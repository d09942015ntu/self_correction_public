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


# data_tags: Which dataset to use,
# data_ratio: How many data should be selected: 1: 100% samples, 0.8: top-80% toxic/non-toxic score, etc.
python3 steering1_build.py --data_tags 1 --data_ratio 1
