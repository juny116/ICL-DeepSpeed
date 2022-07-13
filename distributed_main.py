import logging
import os
import random
import time
import pickle
import json
import csv

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import deepspeed
import numpy as np
import datasets
from datasets import Metric, load_dataset, load_metric, DatasetDict
from tqdm.auto import tqdm
from transformers.deepspeed import HfDeepSpeedConfig
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    start_time = time.time()
    # for debugging purpose
    torch.set_printoptions(profile="full")
    # To avoid warnings about parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  

    set_seed(config['seed'])
    random.seed(config['seed'])

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    ds_config = dict(config['ds_configs'])

    # Process auto config
    if ds_config['zero_optimization']['reduce_bucket_size'] == 'auto':
        ds_config['zero_optimization']['reduce_bucket_size'] = config['models']['hidden_size'] * config['models']['hidden_size']
    if ds_config['zero_optimization']['stage3_prefetch_bucket_size'] == 'auto':
        ds_config['zero_optimization']['stage3_prefetch_bucket_size'] = config['models']['hidden_size'] * config['models']['hidden_size'] * 0.9
    if ds_config['zero_optimization']['stage3_param_persistence_threshold'] == 'auto':
        ds_config['zero_optimization']['stage3_param_persistence_threshold'] = config['models']['hidden_size'] * 10

    # For huggingface deepspeed / Keep this alive!
    dschf = HfDeepSpeedConfig(ds_config)

    # Set logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    datasets.disable_progress_bar()
    if local_rank == 0:
        transformers.logging.set_verbosity_info()
        datasets.logging.set_verbosity_info()
        logger.setLevel(logging.INFO)
    else:
        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()
        logger.setLevel(logging.ERROR)

    # Load data
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    dataset = DatasetDict()
    train_path = config['train_path']
    dataset['train'] = load_dataset('json', data_files=train_path)['train']
    test_path = config['test_path']
    dataset['test'] = load_dataset('json', data_files=test_path)['train']

    def train_preprocess_function(example, acc):
        # Tokenize the texts
        input_sentence = config['experiments']["template"]
        input_sentence = input_sentence.replace('[BOL]', '')
        input_sentence = input_sentence.replace('[S1]', example['sentence1'])
        if "sentence2" in example:
            input_sentence = input_sentence.replace('[S2]', example['sentence2'])

        if acc == 'random':
            # Random Label experiments
            label = random.choice(list(config["experiments"]["verbalizer"].keys()))
        else:
            # Label corruption
            if random.random() <= acc:
                label = example['label']
            else:
                labels = list(config["experiments"]["verbalizer"].keys())
                labels.remove(example['label'])
                label = random.choice(labels)

        input_sentence = input_sentence.replace('[Label]', config["experiments"]["verbalizer"][label])
        example['input_sentence'] = input_sentence
        return example

    def test_preprocess_function(example, idx):
        # Tokenize the texts
        input_sentence = config['experiments']["template"]
        input_sentence = input_sentence.replace('[S1]', example['sentence1'])
        if "sentence2" in example:
            input_sentence = input_sentence.replace('[S2]', example['sentence2'])

        return {"input_sentence": input_sentence, "idx": idx}

    # ensures the main process performs the mapping
    if local_rank > 0:  
        logger.info("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = dataset['train'].map(
        train_preprocess_function,
        desc="Preprocessing dataset...",
        fn_kwargs={'acc': config['demo_accuracy']}
    )
    test_dataset = dataset['test'].map(
        test_preprocess_function,
        desc="Preprocessing dataset...",
        with_indices=True,
    )
    if local_rank == 0:
        torch.distributed.barrier()
            
    demonstrations = [input_sentence for input_sentence in train_dataset['input_sentence']]
    demonstrations = config['experiments']['demo_sep'].join(demonstrations)

    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['models']["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    bol_token = config['models']['bol_token']
    if config['models']['add_bol_to_special_token']:
        tokenizer.add_special_tokens({'additional_special_tokens': [bol_token]})
    bol_token_id = tokenizer.encode(bol_token)[-1]

    # Load model
    logger.info(f'Start loading model {config["models"]["model_name_or_path"]}')
    model_loading_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(config["models"]["model_name_or_path"])
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model_loading_end = time.time()
    logger.info(f'Total time for loading model : {model_loading_end - model_loading_start} sec.')

    batch_size = ds_config['train_micro_batch_size_per_gpu']

    # Evaluate! 
    logger.info("============================ Evaluation ============================")
    logger.info(f"  TASK                                = {config['experiments']['task']}")
    logger.info(f"  Num TRAIN examples                  = {len(train_dataset)}")
    logger.info(f"  Num TEST  examples                  = {len(test_dataset)}")
    logger.info(f"  Demonstration accuracy              = {config['demo_accuracy']}")
    logger.info(f"  Random Seed                         = {config['seed']}")
    logger.info(f"  Inference Model                     = {config['models']['model_name_or_path']}")
    logger.info(f"  Batch szie                          = {batch_size}")
    logger.info(f'======================== in-context samples ========================')
    for line in demonstrations.split('\n'):
        logger.info(line)
    logger.info(f'====================================================================')
    sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(test_dataset, sampler=sampler, batch_size=batch_size)
    metric = load_metric('custom_metric.py', num_process=world_size, process_id=local_rank)

    # only create progressbar for main process
    if local_rank == 0:
        progressbar = tqdm(range(len(dataloader)))
    for step, batch in enumerate(dataloader):
        sentence_inputs = []
        dummy_inputs = []
        current_batch_size = len(batch['input_sentence'])
        for i in range(current_batch_size):
            for label_idx, label_token in config['experiments']['verbalizer'].items():
                sentence_with_label = batch['input_sentence'][i].replace('[Label]', label_token)
                sentence_with_label = demonstrations + config['experiments']['demo_sep'] + sentence_with_label
                dummy_inputs.append(sentence_with_label.replace('[BOL]', bol_token))
                sentence_inputs.append(sentence_with_label.replace('[BOL]', ''))

        inputs = tokenizer(sentence_inputs, padding=True, return_tensors='pt').to(device=local_rank)
        labels = inputs['input_ids']
        token_length = inputs['input_ids'].size(-1)

        dummy_inputs = tokenizer(dummy_inputs, padding=True, return_tensors='pt').to(device=local_rank)
        bol_indices = (dummy_inputs['input_ids'] == bol_token_id).nonzero()[:,1]
        label_masks = [torch.cat((torch.zeros(idx),torch.ones(token_length-idx))) for idx in bol_indices]
        label_masks = torch.stack(label_masks).to(device=local_rank)
        label_masks = label_masks * inputs['attention_mask']

        with torch.no_grad():
            outputs = ds_engine.module(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        label_masks = label_masks[..., 1:].contiguous()

        #### to cpu float32 ####
        labels = labels.cpu().detach()
        label_masks = label_masks.cpu().detach()
        logits = logits.cpu().detach()
        logits = logits.to(torch.float32)

        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses.view(logits.size(0), -1)
        losses = losses * label_masks
        losses = torch.sum(losses, axis=1)
        losses = losses.view(current_batch_size, -1)
        prediction = torch.argmin(losses, dim=1)

        references=[{"idx": batch['idx'][i], "label": batch['label'][i], "probs": losses[i]} for i in range(current_batch_size)]
        metric.add_batch(predictions=prediction, references=references)
        if local_rank == 0:
            progressbar.update(1)

    if local_rank == 0:
        progressbar.close()
    result = metric.compute()
    if local_rank == 0:
        logger.info(f"  ACCURACY                     = {result['accuracy']}")
        logger.info(f"  F1                           = {result['f1']}")
        with open(os.path.join(config['output_path'], f"acc-{config['demo_accuracy']}-seed-{config['seed']}_predictions.jsonl"), 'a') as prediction_file:
            for l, p, prob in zip(result['labels'], result['predictions'], result['probs']):
                json.dump({"label": l, "prediction": p, "prob": prob}, prediction_file)
                prediction_file.write('\n')

    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')
                
if __name__ == "__main__":
    main()
    
    