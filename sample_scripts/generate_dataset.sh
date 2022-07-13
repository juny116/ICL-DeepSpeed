
seeds="13 21 42 87 100"
dataset_path="./data"
n_samples="16"

task_name="cb"
benchmark_name="super_glue"

for seed in $seeds; do
python generate_fewshot.py \
    --task_name $task_name \
    --benchmark_name $benchmark_name \
    --output_dir $dataset_path/$task_name/k-$n_samples-seed-$seed \
    --seed $seed \
    --n_samples $n_samples
done

task_name="sst2"
benchmark_name="glue"

for seed in $seeds; do
python generate_fewshot.py \
    --task_name $task_name \
    --benchmark_name $benchmark_name \
    --output_dir $dataset_path/$task_name/k-$n_samples-seed-$seed \
    --seed $seed \
    --n_samples $n_samples
done

task_name="rte"
benchmark_name="glue"

for seed in $seeds; do
python generate_fewshot.py \
    --task_name $task_name \
    --benchmark_name $benchmark_name \
    --output_dir $dataset_path/$task_name/k-$n_samples-seed-$seed \
    --seed $seed \
    --n_samples $n_samples
done

task_name="trec"
benchmark_name="huggingface"

for seed in $seeds; do
python generate_fewshot.py \
    --task_name $task_name \
    --benchmark_name $benchmark_name \
    --output_dir $dataset_path/$task_name/k-$n_samples-seed-$seed \
    --seed $seed \
    --n_samples $n_samples
done

task_name="ag_news"
benchmark_name="huggingface"

for seed in $seeds; do
python generate_fewshot.py \
    --task_name $task_name \
    --benchmark_name $benchmark_name \
    --output_dir $dataset_path/$task_name/k-$n_samples-seed-$seed \
    --seed $seed \
    --n_samples $n_samples
done
