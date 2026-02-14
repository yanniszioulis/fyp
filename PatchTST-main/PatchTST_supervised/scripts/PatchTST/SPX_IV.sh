if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=21
model_name=PatchTST

root_path_name=./dataset/
data_path_name=SPX_surfaces.csv
model_id_name=SPX_IV
data_name=custom

random_seed=2021
pred_len=63

python3 -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 400 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --fc_dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 100\
  --patience 10\
  --lradj 'TST'\
  --pct_start 0.2\
  --freq 'b'\
  --target 'iv_1.1_1.0' \
  --itr 1 --batch_size 24 --learning_rate 0.0001 >logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log 2>&1
