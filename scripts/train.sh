export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8

model_name=${1}
ctx_len_min=${2}
ctx_len_max=${3}
reg_weight=${4}
lr=${5}
num_passkey=${6}
setting="lr=${lr}-reg=${reg_weight}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}"
exp_name=${model_name}/${setting}
dataset_name=${DATASET_NAME:-datasets/booksum.jsonl.zst}

if [[ ! -f "${dataset_name}" ]]; then
    dataset_name=${BOOKSUM_DATASET_PATH:-datasets/booksum_books_train.jsonl}
    if [[ ! -f "${dataset_name}" ]]; then
        mkdir -p "$(dirname "${dataset_name}")"
        python - "${dataset_name}" <<'PY'
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

output_path = Path(sys.argv[1])
downloaded_path = Path(
    hf_hub_download(
        repo_id="ubaada/booksum-complete-cleaned",
        filename="books/train.jsonl",
        repo_type="dataset",
    )
)
shutil.copyfile(downloaded_path, output_path)
print(f"Prepared BookSum dataset at {output_path}")
PY
    fi
fi

torchrun --nnodes 1 --nproc_per_node 8 \
    duo_attn/train.py \
    --model_name models/${model_name} \
    --batch_size 1 \
    --max_length ${ctx_len_max} \
    --dataset_name "${dataset_name}" \
    --sink_size 128 \
    --recent_size 256 \
    --num_steps 200 \
    --lr ${lr} \
    --reg_weight ${reg_weight} \
    --exp_name $exp_name \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min ${ctx_len_min} \
    --context_length_max ${ctx_len_max} \
    --context_lengths_num_intervals 50 \
    --depth_ratio_num_intervals 1000 \
    --gradient_accumulation_steps 1 \
    --num_passkey ${num_passkey} \
    --dataset_format "multiple_passkey" \
    --output_dir attn_patterns/${exp_name}
