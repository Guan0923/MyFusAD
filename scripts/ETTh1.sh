#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p run_logs
mkdir -p checkpoints_grid

PRED_LENS=(96)

SEQ_LENS=(336)
PATCH_SIZES=(24)
MASK_RATIOS=(0.25)
KERNEL_SIZES=(3)
BATCH_SIZES=(128)
EMB_DIMS=(64)

SEEDS=(42 43 44 45 46)

DEPTH=3
DROPOUT=(0.1 0.2 0.3 0.4 0.5)
TRAIN_EPOCHS=50
PRETRAIN_EPOCHS=20

for pred_len in "${PRED_LENS[@]}"
do
  for seq_len in "${SEQ_LENS[@]}"
  do
    for patch_size in "${PATCH_SIZES[@]}"
    do
      for mask_ratio in "${MASK_RATIOS[@]}"
      do
        for kernel_size in "${KERNEL_SIZES[@]}"
        do
          for batch_size in "${BATCH_SIZES[@]}"
          do
            for emb_dim in "${EMB_DIMS[@]}"
            do
              for seed in "${SEEDS[@]}"
              do
                for dropout in "${DROPOUT[@]}"
                do
                  mr_tag=${mask_ratio//./p}
                  dp_tag=${dropout//./p}

                  exp_name="ETTh1_pl${pred_len}_seq${seq_len}_ps${patch_size}_mr${mr_tag}_k${kernel_size}_bs${batch_size}_emb${emb_dim}_dp${dp_tag}_seed${seed}"
                  log_file="run_logs/${exp_name}.log"
                  ckpt_dir="checkpoints_grid/${exp_name}"

                  mkdir -p "${ckpt_dir}"

                  {
                    echo "============================================================"
                    echo "Running ${exp_name}"
                    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
                    echo "Checkpoint dir: ${ckpt_dir}"
                    echo "============================================================"

                    python -u FusAD_Forecasting.py \
                      --root_path ./data/ETT-small \
                      --data ETTh1 \
                      --data_path ETTh1.csv \
                      --features M \
                      --target OT \
                      --freq h \
                      --seq_len ${seq_len} \
                      --pred_len ${pred_len} \
                      --emb_dim ${emb_dim} \
                      --depth ${DEPTH} \
                      --batch_size ${batch_size} \
                      --dropout ${dropout} \
                      --patch_size ${patch_size} \
                      --mask_ratio ${mask_ratio} \
                      --train_epochs ${TRAIN_EPOCHS} \
                      --pretrain_epochs ${PRETRAIN_EPOCHS} \
                      --ifm_kernel_size ${kernel_size} \
                      --ASM True \
                      --IFM True \
                      --adaptive_filter True \
                      --load_from_pretrained True \
                      --seed ${seed}

                    echo "============================================================"
                    echo "Finished ${exp_name}"
                    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
                    echo "============================================================"
                  } 2>&1

                done
              done
            done
          done
        done
      done
    done
  done
done