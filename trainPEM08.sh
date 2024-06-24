# exp_dir="logs/baseLine/PEM08/"
batch_size=16
lr=0.0001
weight_decay=$WEIGHT_DECALY
seq_len=12
pre_len=12
dropout=0.0
dataset="PEMS08"
exp_dir="logs/$ExpName/$dataset/"

python train.py --dataset $dataset --epochs $epochs \
--batchsize $batch_size --lr $lr --weight_decay $weight_decay \
--seq_len $seq_len --pre_len $pre_len --dropout $dropout \
--val_test_data $VAL_TEST_DATA \
--multi_size_t --lstm  lstm  \
--exp_name MSST --exp_dir $exp_dir;
