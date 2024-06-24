batch_size=16
lr=0.002
# lr=0.0004
weight_decay=$WEIGHT_DECALY
seq_len=12
pre_len=12
dropout=0.0
dataset="PEMS03"
exp_dir="logs/$ExpName/$dataset/"

# python train.py --dataset $dataset --epochs $epochs \
# --batchsize $batch_size --lr $lr --weight_decay $weight_decay \
# --val_test_data $VAL_TEST_DATA \
# --seq_len $seq_len --pre_len $pre_len --dropout $dropout \
# --lstm  None  \
# --exp_name PEMS3_BS --exp_dir $exp_dir;

# python train.py --dataset $dataset --epochs $epochs \
# --batchsize $batch_size --lr $lr --weight_decay 1e-4 \
# --seq_len $seq_len --pre_len $pre_len --dropout $dropout \
# --multi_size_t --lstm  lstm  \
# --exp_name PEMS3_lr4e3_MS_lstm_wd10e-5 --exp_dir $exp_dir;

# python train.py --dataset $dataset --epochs $epochs \
# --batchsize $batch_size --lr $lr --weight_decay 1.2e-4 \
# --seq_len $seq_len --pre_len $pre_len --dropout $dropout \
# --multi_size_t --lstm  lstm  \
# --exp_name PEMS3_lr4e3_MS_lstm12e-5 --exp_dir $exp_dir;


python train.py --dataset $dataset --epochs $epochs \
--batchsize $batch_size --lr $lr --weight_decay $weight_decay \
--seq_len $seq_len --pre_len $pre_len --dropout $dropout \
--val_test_data $VAL_TEST_DATA \
--multi_size_t --lstm  lstm  \
--exp_name PEMS3_MSST --exp_dir $exp_dir;
