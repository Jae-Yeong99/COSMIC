#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ==============================
# Basic settings
# ==============================
data_root="/data"

# ImageNet-A / ImageNet-R only
testsets="A/R"

ctx_init="a_photo_of_a"

arch="ViT-B/16"          # RN50 | ViT-B/16 | align-base
dino_size="l"            # l | b | s
center_type_clip="attn"  # default | ema | attn
center_type_dino="default"

gpu_id=1
batch_size=8

current_time="$(date +"%Y_%m_%d_%H_%M_%S")"

# ==============================
# Sanity checks
# ==============================
echo "[INFO] data_root      = ${data_root}"
echo "[INFO] testsets       = ${testsets}"
echo "[INFO] arch           = ${arch}"
echo "[INFO] gpu            = ${gpu_id}"

# dataset existence check
for d in imagenet-a imagenet-r; do
  if [ ! -d "${data_root}/${d}" ]; then
    echo "[ERROR] Missing dataset directory: ${data_root}/${d}"
    exit 1
  fi
done

# ==============================
# Run COSMIC
# ==============================
python3 ./cosmic_main.py "${data_root}" \
  --test_sets "${testsets}" \
  -a "${arch}" \
  --ctx_init "${ctx_init}" \
  --text_prompt tip_cupl \
  --gpu "${gpu_id}" \
  --beta 5.5 \
  --config configs_l \
  --seed 0 \
  --selection_p 0.1 \
  -b "${batch_size}" \
  --DINO_size "${dino_size}" \
  --center_type_clip "${center_type_clip}" \
  --center_type_dino "${center_type_dino}" \
  --r 0.2 \
  --DINO_Cache_shot 16 \
  --CLIP_Cache_shot 16 \
  --use_clip_cache \
  --DINOv2 \
  --use_clip_clique \
  --use_dino_clique \
  --mac_step 100 \
  --target_avg_degree 5.0 \
  --inrease_t \
  -p 1000 \
  --log_time "${current_time}" \
  --clip_is_DMN \
  --is_SOG \
  --use_log
