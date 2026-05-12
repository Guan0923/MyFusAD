# 192 336 720
for len in 96 
do
  python -u FusAD_Forecasting.py \
  --root_path ./data/ETT-small \
  --pred_len $len \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 3 \
  --batch_size 512 \
  --dropout 0.5 \
  --patch_size 24 \
  --train_epochs 20 \
  --pretrain_epochs 10 \
  --ASM True \
  --IFM True \
  --adaptive_filter True
done
