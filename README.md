# In-context Learning


## How to Instsall

1. <code>pip install -r requirements.txt</code> to install additional libraries.

2. Install deepspeed

## How to Run (example)
See details about config in conf directory

<code> deepspeed --include localhost:0,1,2,3 --no_local_rank distributed_main.py ds_configs=zero3 experiments=sst2 models=gpt-j seed=100 </code>

## Notice!
* Few-shot data file is responsible for **Order** & **Label Balance** of demonstrations
    - Process order & balance in label sampling stage
    - Check sample few-shot data for details

* Experiment config manages **Templates**, **Verbalizers**, **Methods**, etc.
    - Templates include instructions and prompts
    - Methods include infernece method(direct or channel)

* Deepspeed config manages experiment environments including dtype, visible gpus, zero stage, etc.

## Sampling Random Dataset
Run <code>generated_fewshot.py</code> to randomly sample datasets. We generate <code>train.jsonl</code> for train set and <code>test.jsonl</code> for test set. (We just copy the original dataset for test set, only the formatting changes.)
Datasets are saved in json format for each sample.
- <code>label</code> : label of the sample
- <code>sentence1</code> : first input of the sample.
- <code>sentence2</code> : second input of the sample. For single-sentence tasks, <code>sentence2</code> is not given.

Parameters
1. <code>task_name</code>
2. <code>benchmark_name</code> : select from <code>glue</code>, <code>super_glue</code>, <code>tweet_eval</code>, <code>huggingface</code>
    - <code>huggingface</code> is for tasks without any specific benchmark (e.g., <code>trec</code>, <code>ag_news</code>)
3. <code>output_dir</code> 
4. <code>seed</code> : random seed
5. <code>n_samples</code> : number of samples (= k)
6. <code>balance</code> : if given, we sample equal number of samples per class

To run sample scripts : <code>sh sample_scripts/generate_dataset.sh</code>

