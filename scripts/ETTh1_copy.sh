# 96 192 336 720
for len in 96
do
  python -u FusAD_Forecasting.py \
    --root_path ./data/ETT-small \
    --pred_len $len \
    --data ETTh1 \
    --data_path ETTh1.csv \
    --seq_len 336 \
    --emb_dim 64 \
    --depth 3 \
    --batch_size 512 \
    --dropout 0.5 \
    --patch_size 24 \
    --train_epochs 50 \
    --pretrain_epochs 20 \
    --ifm_kernel_size 3 \
    --ASM True \
    --IFM True \
    --adaptive_filter True
done
