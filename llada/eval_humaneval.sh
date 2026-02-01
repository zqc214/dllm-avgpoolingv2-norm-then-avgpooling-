# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=512
block_length=32
# steps=$((length / block_length))
steps=512

# Dual cache parameters
prefix_keep_ratio=0.5
suffix_keep_ratio=0.5
keep_first_n=5
keep_last_n=5

# # baseline
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
# --output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# # prefix cache
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True \
# --output_path evals_results/prefix_cache/humaneval-ns0-${length} --log_samples

# # parallel
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True \
# --output_path evals_results/parallel/humaneval-ns0-${length} --log_samples

# # prefix cache+parallel
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
# --output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples

# dual cache+parallel
CUDA_VISIBLE_DEVICES=6 accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,prefix_keep_ratio=${prefix_keep_ratio},suffix_keep_ratio=${suffix_keep_ratio},keep_first_n=${keep_first_n},keep_last_n=${keep_last_n},show_speed=True \
--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length}-pr${prefix_keep_ratio}-sr${suffix_keep_ratio}-f${keep_first_n}-l${keep_last_n} --log_samples

# ## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}
