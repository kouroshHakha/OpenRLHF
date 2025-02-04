set -x 

export HF_HUB_ENABLE_HF_TRANSFER=1
[ -z "${WANDB_API_KEY}" ] && { echo "Error: WANDB_API_KEY is not set"; exit 1; }

python3 -m openrlhf.cli.train_ppo_ray \
   --advantage_estimator reinforce \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain Qwen/Qwen2.5-math-7B \
   --save_steps 10 \
   --save_path /mnt/local_storage/openrlhf/checkpoint/Qwen/Qwen2.5-math-7B-gsm8k/final \
   --save_hf_ckpt \
   --ckpt_path /mnt/local_storage/openrlhf/checkpoint/Qwen/Qwen2.5-math-7B-gsm8k/ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 40 \
   --max_samples 256 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 3072 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data /home/ray/default/data/math_train_data_processed_with_qwen_prompt.json \
   --input_key input \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb $WANDB_API_KEY \
   --wandb_project openrlhf \
   --remote_rm_url http://localhost:5000/get_reward_math

   # --apply_chat_template \

   # --packing_samples \

   # --pretrain meta-llama/Llama-3.1-8B-Instruct \


   # --reward_num_nodes 1 \
   # --reward_num_gpus_per_node 2 \
   # --critic_num_nodes 1 \
   # --critic_num_gpus_per_node 2 \

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)