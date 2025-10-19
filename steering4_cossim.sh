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

python3 steering4_cossim.py --steering_vec="outputs/Qwen2.5-3B-Instruct/steering_d1_t100/1_vector/steer.json" --limit=500
