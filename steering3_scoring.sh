set -e
source venv/bin/activate

DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE

data_types=("toxic" "non_toxic")
hook_types=("hook_seq" "hook_random")

for hook_type in ${hook_types[@]}; do
  for data_type in ${data_types[@]}; do
    python3 steering3_scoring.py --input_dir="outputs/gemma-3-4b-it/steering_d1_t100/2_inference/${hook_type}/${data_type}" --limit=1000
  done
done

