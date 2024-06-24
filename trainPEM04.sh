# exp_dir="logs/baseLine/PEM04/"
batch_size=16
lr=0.001
weight_decay=9e-5
seq_len=12
pre_len=12
dropout=0.1
dataset="PEMS04"
exp_dir="logs/$ExpName/$dataset/"
# python train.py --dataset $dataset --epochs $epochs \
# --batchsize $batch_size --lr $lr --weight_decay $weight_decay \
# --val_test_data $VAL_TEST_DATA \
# --seq_len $seq_len --pre_len $pre_len --dropout $dropout \
# --lstm  None  \
# --exp_name BS --exp_dir $exp_dir;

python train.py --dataset $dataset --epochs $epochs \
--batchsize $batch_size --lr $lr --weight_decay $weight_decay \
--seq_len $seq_len --pre_len $pre_len --dropout $dropout \
--val_test_data $VAL_TEST_DATA \
--multi_size_t --lstm  lstm  \
--exp_name MSST --exp_dir $exp_dir;