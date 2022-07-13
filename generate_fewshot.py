import argparse
import os
import random
import time
import json

from datasets import load_dataset
from tqdm.auto import tqdm

from transformers import (
    set_seed,
)

task2keys = {
    'sst2' : {
        'input_key' : ('sentence', None),
        'label_key' : 'label'
    },
    'trec' : {
        'input_key' : ('text', None),
        'label_key' : 'label-coarse'
    },
    'ag_news' : {
        'input_key' : ('text', None),
        'label_key' : 'label'
    },
    'hate' : {
        'input_key' : ('text', None),
        'label_key' : 'label'
    },
    'cb' : {
        'input_key' : ('premise', 'hypothesis'),
        'label_key' : 'label'
    },
    'rte' : {
        'input_key' : ('sentence1', 'sentence2'),
        'label_key' : 'label'
    },
    'mnli' : {
        'input_key' : ('premise', 'hypothesis',),
        'label_key' : 'label'
    },
    'mrpc' : {
        'input_key' : ('sentence1', 'sentence2',),
        'label_key' : 'label'
    },
    'sick' : {
        'input_key' : ('sentence_A', 'sentence_B',),
        'label_key' : 'label'
    },
    'poem_sentiment' : {
        'input_key' : ('verse_text', None,),
        'label_key' : 'label'
    },
    # only train set
    'medical_questions_pairs' : {
        'input_key' : ('question_1', 'question_2',),
        'label_key' : 'label',
        'train_size' : 2438,
        'validation_size' : 610
    },
    # glue
    'wnli' : {
        'input_key' : ('sentence1', 'sentence2',),
        'label_key' : 'label'
    },
    # only test set
    'climate_fever' : {
        'input_key' : ('claim', None,),
        'label_key' : 'claim_label',
        'train_size' : 1228,
        'validation_size' : 307
    },
    'sick' : {
        'input_key' : ('sentence_A', 'sentence_B',),
        'label_key' : 'label'
    },
    'hate_speech18' : {
        'input_key' : ('text', None,),
        'label_key' : 'label',
        'train_size' : 8562,
        'validation_size' : 2141
    },
    # ethos
    # only train set
    'national_origin' : {
        'input_key' : ('text', None,),
        'label_key' : 'national_origin',
        'train_size' : 346,
        'validation_size' : 87
    },
    'race' : {
        'input_key' : ('text', None,),
        'label_key' : 'race',
        'train_size' : 346,
        'validation_size' : 87
    },
    'religion' : {
        'input_key' : ('text', None,),
        'label_key' : 'religion',
        'train_size' : 346,
        'validation_size' : 87
    },
    # tweet_eval
    'stance_atheism' : {
        'input_key' : ('text', None,),
        'label_key' : 'label'
    },
    'stance_feminist' : {
        'input_key' : ('text', None,),
        'label_key' : 'label'
    },
    # financial_phrasebank
    'sentences_allagree' : {
        'input_key' : ('sentence', None,),
        'label_key' : 'label',
        'train_size' : 1811,
        'validation_size' : 453
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the task to generate few-shot.",
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="The name of the benchmark to generate few-shot.",
        choices=['glue', 'super_glue', 'huggingface', 'tweet_eval', 'ethos', 'financial_phrasebank'],
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=100, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=0, 
        help="Number of samples for in-context learning."
    )
    # for balanced sampling
    parser.add_argument(
        '--balance', 
        default=False, 
        action="store_true",
        help='Balance samples per label for in-context learning.'
    )
    args = parser.parse_args() 

    return args


def main():
    args = parse_args()

    stats = dict()
    stats['task_name'] = args.task_name
    stats['benchmark_name'] = args.benchmark_name
    stats['seed'] = args.seed
    stats['n_samples'] = args.n_samples
    stats['balance'] = args.balance

    print(f'Generate dataset samples to path : {args.output_dir}')
    # mkdir output directory to save logs and configs.
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # set random seet to 1 to sample same train/validation set 
    set_seed(1)
    random.seed(1)

    # key for input sentence
    keys_dict = task2keys.get(args.task_name)
    sentence1_key, sentence2_key = keys_dict.get('input_key')
    label_key = keys_dict.get('label_key')
    if 'train_size' in keys_dict.keys():
        train_size = keys_dict.get('train_size')
        validation_size = keys_dict.get('validation_size')

    train_dataset = None
    test_dataset = None
    # load dataset
    if args.benchmark_name == 'huggingface':
        # TREC, AGNews
        if args.task_name in ['sick', 'poem_sentiment', 'sick']:
            test_dataset = load_dataset(args.task_name, split='validation')
        elif args.task_name in ['climate_fever']:
            # no train set, split test set.
            print(f'No train set for {args.task_name}. Split test set {train_size} / {validation_size}')
            dataset = load_dataset(args.task_name, split='test')
            dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
            train_dataset, test_dataset = dataset['train'], dataset['test']

            # train_dataset = load_dataset(args.task_name, split=f'test[:{train_size}]')
            # test_dataset = load_dataset(args.task_name, split=f'test[-{validation_size}:]')
        elif args.task_name in ['medical_questions_pairs', 'hate_speech18']:
            # no validation set, split train set.
            print(f'No validation set for {args.task_name}. Split train set {train_size} / {validation_size}')
            dataset = load_dataset(args.task_name, split='train')
            if args.task_name == 'hate_speech18':
                dataset = dataset.filter(lambda example : example[label_key] in [0, 1])
            dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
            train_dataset, test_dataset = dataset['train'], dataset['test']

            # train_dataset = load_dataset(args.task_name, split=f'train[:{train_size}]')
            # test_dataset = load_dataset(args.task_name, split=f'train[-{validation_size}:]')
        else:
            test_dataset = load_dataset(args.task_name, split='test')

        if train_dataset is None:
            train_dataset = load_dataset(args.task_name, split='train')
    elif args.benchmark_name == 'ethos':
        dataset = load_dataset(args.benchmark_name, 'multilabel', split='train')
        dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']

        # train_dataset = load_dataset(args.benchmark_name, 'multilabel', split=f'train[:{train_size}]')
        # test_dataset = load_dataset(args.benchmark_name, 'multilabel', split=f'train[-{validation_size}:]')
    elif args.benchmark_name == 'financial_phrasebank':
        dataset = load_dataset(args.benchmark_name, args.task_name, split='train')
        dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size, shuffle=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']

        # train_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'train[:{train_size}]')
        # test_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'train[-{validation_size}:]')

    else:
        # GLUE, SuperGLUE, tweet_eval
        train_dataset = load_dataset(args.benchmark_name, args.task_name, split='train')
        if args.task_name == 'mnli':
            test_dataset = load_dataset(args.benchmark_name, args.task_name, split='validation_matched')
        else:
            test_dataset = load_dataset(args.benchmark_name, args.task_name, split='validation')

    

    # filter labels
    test_label_set = set(test_dataset[label_key])
    print('Filtering unused labels...')
    before_len = len(train_dataset)
    train_dataset = train_dataset.filter(lambda example : example[label_key] in test_label_set)
    after_len = len(train_dataset)
    print(f'Filtered train set size : {before_len} -> {after_len}')

    # assert
    train_label_count = len(set(train_dataset[label_key]))
    test_label_count = len(set(test_dataset[label_key]))
    assert train_label_count == test_label_count, f'Train label count({train_label_count}) != test label count ({test_label_count})'

    # used for balanced sampling
    num_class = train_label_count

    print('======== Split Stat ========')
    print(f'Train     : {len(train_dataset)}')
    print(f'Eval      : {len(test_dataset)}')
    print(f'Num class : {num_class}')

    # train_dataset_per_class_list = [train_dataset.filter(lambda sample: sample[label_key] == class_index) for class_index in range(num_class)]
    
    train_dataset = [sample for sample in train_dataset]

    print('TRAIN ========================')
    # split train set for balanced selection
    train_dataset_per_class_list = [list(filter(lambda sample: sample[label_key] == class_index, train_dataset)) for class_index in range(num_class)]
    for class_index, train_dataset_per_class in enumerate(train_dataset_per_class_list):
        print(f'# class {class_index} : {len(train_dataset_per_class)}')

    print('TEST ========================')
    # split train set for balanced selection
    test_dataset_per_class_list = [list(filter(lambda sample: sample[label_key] == class_index, test_dataset)) for class_index in range(num_class)]
    for class_index, test_dataset_per_class in enumerate(test_dataset_per_class_list):
        print(f'# class {class_index} : {len(test_dataset_per_class)}')
    
     
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    ## SAVE TRAIN SET ##
    train_stats = dict()
    selected_samples = []
    # select TRAIN samples
    for i in tqdm(range(args.n_samples), desc="Sampling training set"):
        sample_dict = dict()

        if args.balance:
            class_index = i % num_class

            train_dataset_per_class = train_dataset_per_class_list[class_index]
            train_count = len(train_dataset_per_class)
            random_index = random.randint(0, train_count-1)
            random_sample = train_dataset_per_class.pop(random_index)

            train_dataset_per_class_list[class_index] = train_dataset_per_class
        else:
            train_count = len(train_dataset)
            random_index = random.randint(0, train_count-1)
            random_sample = train_dataset.pop(random_index)

        sentence1 = random_sample[sentence1_key]
        sentence2 = random_sample[sentence2_key] if sentence2_key else None
        label = random_sample[label_key]

        sample_dict['sentence1'] = sentence1
        if sentence2:
            sample_dict['sentence2'] = sentence2
        sample_dict['label'] = label

        selected_samples.append(sample_dict)

        # for config
        train_stats[label] = train_stats.get(label, 0) + 1
    
    # write to output file
    train_path = os.path.join(args.output_dir, 'train.jsonl')
    with open(train_path, 'w') as output_file:
        # we shuffle the samples in balanced setting
        # TODO : other configs for balanced setting?
        # if args.balance:
        #     random.shuffle(selected_samples)

        for selected_sample in selected_samples:
            output_file.write(json.dumps(selected_sample, ensure_ascii=False) + '\n')

    ## SAVE TEST SET ##
    # copy test set in the same format
    test_stats = dict()
    test_path = os.path.join(args.output_dir, 'test.jsonl')
    test_count = len(test_dataset)
    with open(test_path, 'w') as output_file:
        for index in tqdm(range(test_count), desc="Sampling test set"):
            sample_dict = dict()
            test_sample = test_dataset[index]

            sentence1 = test_sample[sentence1_key]
            sentence2 = test_sample[sentence2_key] if sentence2_key else None
            label = test_sample[label_key]

            sample_dict['sentence1'] = sentence1
            if sentence2:
                sample_dict['sentence2'] = sentence2
            sample_dict['label'] = label

            output_file.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')


            # for config
            test_stats[label] = test_stats.get(label, 0) + 1

    
    stats['train_stats'] = train_stats
    stats['test_stats'] = test_stats

    stats_path = os.path.join(args.output_dir, 'stats.json')
    with open(stats_path, 'w') as file:
        file.write(json.dumps(stats))

if __name__ == "__main__":
    print('Generate few-shot data.py')
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total runtime : {end_time - start_time} sec.')