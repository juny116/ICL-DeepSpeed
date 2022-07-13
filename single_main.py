import logging
import os
import random
import time
import csv
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from datasets import load_dataset, load_metric, DatasetDict
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

dtype_dict = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    start_time = time.time()
    # for debugging purpose
    torch.set_printoptions(profile="full")

    device = torch.device(config['ds_configs']['device'])
    dtype = dtype_dict[config['ds_configs']['dtype']]

    # Set default seed 100 just in case! (e.g. model bias?)
    set_seed(100)
    random.seed(100)

    logger = logging.getLogger(__name__)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)

    # Open result writer
    result_file = open(os.path.join(config['output_path'], "results.tsv"), "a")
    result_writer = csv.writer(result_file, delimiter='\t')
    result_writer.writerow(['demo_accuracy', 'seed', 'accuracy', 'f1'])

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
    model = AutoModelForCausalLM.from_pretrained(config["models"]["model_name_or_path"], torch_dtype=dtype).to(device)
    model.eval()
    model_loading_end = time.time()
    logger.info(f'Total time for loading model : {model_loading_end - model_loading_start} sec.')

    for acc in config['demo_accuracy']:
        for seed in config['seeds']:
            # Set seed
            set_seed(seed)
            random.seed(seed)

            # Load data
            # In distributed training, the load_dataset function guarantee that only one local process can concurrently
            dataset = DatasetDict()
            train_path = f"{config['data_path']}/{config['experiments']['task']}/k-{config['k']}-seed-{seed}/train.jsonl"
            dataset['train'] = load_dataset('json', data_files=train_path)['train']
            test_path = f"{config['data_path']}/{config['experiments']['task']}/test.jsonl"
            dataset['test'] = load_dataset('json', data_files=test_path)['train']

            if acc == 'random':
                label_set = list(config["experiments"]["verbalizer"].keys())
            if acc == 'shuffle':
                label_set = [item['label'] for item in dataset['train']]
                random.shuffle(label_set)

            def train_preprocess_function(example):
                # Tokenize the texts
                input_sentence = config['experiments']["template"]
                input_sentence = input_sentence.replace('[BOL]', '')
                input_sentence = input_sentence.replace('[S1]', example['sentence1'])
                if "sentence2" in example:
                    input_sentence = input_sentence.replace('[S2]', example['sentence2'])

                if acc == 'random':
                    # Random Label experiments
                    label = random.choice(label_set)
                elif acc == 'shuffle':
                    # Shuffle Label experiments
                    label = label_set.pop()
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
    
            train_dataset = dataset['train'].map(
                train_preprocess_function,
                desc="Preprocessing dataset...",
            )
            test_dataset = dataset['test'].map(
                test_preprocess_function,
                desc="Preprocessing dataset...",
                with_indices=True
            )

            demonstrations = [input_sentence for input_sentence in train_dataset['input_sentence']]
            demonstrations = config['experiments']['demo_sep'].join(demonstrations)

            batch_size = config['ds_configs']['batch_size']

            # Evaluate! 
            logger.info("***** Zero/Few-shot Evaluation *****")
            logger.info(f"  TASK                                = {config['experiments']['task']}")
            logger.info(f"  Num TRAIN examples                  = {len(train_dataset)}")
            logger.info(f"  Num TEST  examples                  = {len(test_dataset)}")
            logger.info(f"  Random Seed                         = {seed}")
            logger.info(f"  Demo accuracy                       = {acc}")
            logger.info(f"  Inference Model                     = {config['models']['model_name_or_path']}")
            logger.info(f'=== in-context samples ===\n{demonstrations}\n=====================')

            dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
            metric = load_metric('custom_metric.py')


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

                inputs = tokenizer(sentence_inputs, padding=True, return_tensors='pt').to(device)
                labels = inputs['input_ids']
                token_length = inputs['input_ids'].size(-1)

                dummy_inputs = tokenizer(dummy_inputs, padding=True, return_tensors='pt').to(device)
                bol_indices = (dummy_inputs['input_ids'] == bol_token_id).nonzero()[:,1]
                label_masks = [torch.cat((torch.zeros(idx),torch.ones(token_length-idx))) for idx in bol_indices]
                label_masks = torch.stack(label_masks).to(device)
                label_masks = label_masks * inputs['attention_mask']

                with torch.no_grad():
                    outputs = model(**inputs)
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

                progressbar.update(1)

                references=[{"idx": batch['idx'][i], "label": batch['label'][i], "probs": losses[i]} for i in range(current_batch_size)]
                metric.add_batch(predictions=prediction, references=references)

            result = metric.compute()

            logger.info(f"  ACCURACY                     = {result['accuracy']}")
            logger.info(f"  F1                           = {result['f1']}")
            result_writer.writerow([acc, seed, result['accuracy'], result['f1']])
            with open(os.path.join(config['output_path'], f"acc-{acc}-seed-{seed}_predictions.jsonl"), 'a') as prediction_file:
                for l, p, prob in zip(result['labels'], result['predictions'], result['probs']):
                    json.dump({"label": l, "prediction": p, "prob": prob}, prediction_file)
                    prediction_file.write('\n')

    if config['zero-shot']:
        logger.info("***** Zero-shot Evaluation *****")
        logger.info(f"  TASK                                = {config['experiments']['task']}")
        logger.info(f"  Num TEST  examples                  = {len(test_dataset)}")
        logger.info(f"  Inference Model                     = {config['models']['model_name_or_path']}")

        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        metric = load_metric('custom_metric.py')

        progressbar = tqdm(range(len(dataloader)))
        for step, batch in enumerate(dataloader):
            sentence_inputs = []
            dummy_inputs = []
            current_batch_size = len(batch['input_sentence'])
            for i in range(current_batch_size):
                for label_idx, label_token in config['experiments']['verbalizer'].items():
                    sentence_with_label = batch['input_sentence'][i].replace('[Label]', label_token)
                    dummy_inputs.append(sentence_with_label.replace('[BOL]', bol_token))
                    sentence_inputs.append(sentence_with_label.replace('[BOL]', ''))

            inputs = tokenizer(sentence_inputs, padding=True, return_tensors='pt').to(device)
            labels = inputs['input_ids']
            token_length = inputs['input_ids'].size(-1)

            dummy_inputs = tokenizer(dummy_inputs, padding=True, return_tensors='pt').to(device)
            bol_indices = (dummy_inputs['input_ids'] == bol_token_id).nonzero()[:,1]
            label_masks = [torch.cat((torch.zeros(idx),torch.ones(token_length-idx))) for idx in bol_indices]
            label_masks = torch.stack(label_masks).to(device)
            label_masks = label_masks * inputs['attention_mask']

            with torch.no_grad():
                outputs = model(**inputs)
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

            progressbar.update(1)

            references=[{"idx": batch['idx'][i], "label": batch['label'][i], "probs": losses[i]} for i in range(current_batch_size)]
            metric.add_batch(predictions=prediction, references=references)

        result = metric.compute()

        logger.info(f"  ACCURACY                     = {result['accuracy']}")
        logger.info(f"  F1                           = {result['f1']}")
        result_writer.writerow(["None", "None", result['accuracy'], result['f1']])
        with open(os.path.join(config['output_path'], f"zero-shot_predictions.jsonl"), 'a') as prediction_file:
            for l, p, prob in zip(result['labels'], result['predictions'], result['probs']):
                json.dump({"label": l, "prediction": p, "prob": prob}, prediction_file)
                prediction_file.write('\n')

    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')

if __name__ == "__main__":
    main()
