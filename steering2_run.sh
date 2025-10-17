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


s_dirs=(-1 1) # -1: Negative Steering, 1: Positive Steering
#s_layers=(-1 1 2 4 6 8 10 12 14 16 18 20 23 26 29 32)
s_layers=(-1 1 6 12 18 26 32 2 4 8 10 14 16 20 23 29) # -1: No Steering Vector, 1: 1th-Layer
limits=(384 768 1000) # How many samples should be run
hook_types=("hook_seq" "hook_random") # hook_seq: steering vector over sequence of tokens, hook_random: random steering vector
data_types=("toxic" "non_toxic") # whether initial prompt is toxic or non-toxic

for limit in ${limits[@]}; do
  for s_layer in ${s_layers[@]};do
    for data_type in ${data_types[@]};do
      for hook_type in ${hook_types[@]};do
        for s_dir in ${s_dirs[@]}; do
          python3 steering2_run.py --steering_vec="outputs/gemma-3-4b-it/steering_d1_t100/1_vector/steer.json"  \
              --limit=${limit}  \
              --s_layer=${s_layer} \
              --data_type=${data_type} \
              --hook_type=${hook_type} \
              --s_dir=${s_dir}
        done
      done
    done
  done
done
