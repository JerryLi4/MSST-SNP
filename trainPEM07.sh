# exp_dir="logs/baseLine/PEMS07"
batch_size=8
lr=0.0001
weight_decay=7e-5
seq_len=12
pre_len=12
dropout=0.0
dataset="PEMS07"
exp_dir="logs/$ExpName/$dataset/"

python train.py --dataset $dataset --epochs $epochs \
--batchsize $batch_size --lr $lr --weight_decay $weight_decay \
--val_test_data $VAL_TEST_DATA \
--seq_len $seq_len --pre_len $pre_len --dropout $dropout \
--multi_size_t --lstm  lstm  \
--exp_name PEMS7_lr4e3_MS_lstm --exp_dir $exp_dir;

