import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


from Dataloader import *
from model import MSST_SNP
from opt import WarmUpScheduler
from utils import metric
from utils import TrainTimer
from utils import Logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="PEMS03", help="the datasets name")
parser.add_argument('--exp_name', type=str, default="snp", help="the datasets name")
parser.add_argument('--exp_dir', type=str, default="./result", help="the datasets name")
parser.add_argument('--val_test_data', type=bool, default=False, help="the datasets name")

parser.add_argument('--multi_size_t', action='store_true', help="enable multi-scale temporal")
parser.add_argument('--lstm', type=str, default="lstm", help="lstm type. None | lstm | lstm_snp")
parser.add_argument('--attention_snp', action='store_true', help="enable attention_snp")

parser.add_argument('--train_rate', type=float, default=0.6, help="The ratio of training set")
parser.add_argument('--seq_len', type=int, default=24, help="The length of input sequence")
parser.add_argument('--pre_len', type=int, default=24, help="The length of output sequence")
parser.add_argument('--batchsize', type=int, default=1, help="Number of training batches")
parser.add_argument('--heads', type=int, default=4, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=7e-5, help="Learning rate")
parser.add_argument('--in_dim', type=float, default=1, help="Dimensionality of input data")
parser.add_argument('--embed_size', type=float, default=64, help="Embed_size")
parser.add_argument('--epochs', type=int, default=1, help="epochs")
parser.add_argument('--warm_up', type=bool, default=True, help="Enable warm_up")
args = parser.parse_args()

if __name__ == "__main__":
    # exp_dir = os.path.join(args.exp_dir, args.exp_name)
    exp_dir = os.path.join(args.exp_dir)
    if os.path.exists(exp_dir) == False:
        os.makedirs(exp_dir)
    log_save_dir = f"{exp_dir}/{args.exp_name}_{ args.pre_len * 5}min_{args.dataset}_mean_std.txt"
    print(f"Experment directory: {exp_dir}")

    model_pkl_path = os.path.join(exp_dir, f"{args.exp_name}_{args.dataset}")
    if os.path.exists(model_pkl_path) == False:
        os.makedirs(model_pkl_path)
    best_model_pkl_path = os.path.join(model_pkl_path, f"best_mae.pkl")

    log = Logger(log_save_dir)
    data, adj = load_data(args.dataset)
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    data1 = np.mat(data, dtype=np.float32)
    trainX, trainY, valX, valY, testX, testY, mean, std = preprocess_data(data1, time_len, args.train_rate,
                                                                          args.seq_len, args.pre_len)
    train_data = TensorDataset(trainX, trainY)
    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    if args.val_test_data:
        val_data = TensorDataset(testX, testY)
        test_data = TensorDataset(valX, valY)
    else:
        val_data = TensorDataset(valX, valY)
        test_data = TensorDataset(testX, testY)
    val_dataloader = DataLoader(val_data, batch_size=args.batchsize)
    test_data = TensorDataset(valX,valY)
    test_dataloader = DataLoader(test_data,batch_size=args.batchsize)
    train_timer = TrainTimer(args.epochs, len(train_dataloader), args.batchsize)
    

    model = MSST_SNP(adj, args.in_dim, args.embed_size,
                  args.seq_len, args.pre_len, args.heads, 4, args.dropout, configs = args)
    # log.info(summary(model, input_size=(args.batchsize, args.seq_len, num_nodes, args.in_dim)))
    model = model.to(device)
    criterion = nn.MSELoss()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                   len_loader=len(train_dataloader),
                                   warmup_steps=len(train_dataloader) * args.epochs * 0.05,
                                   warmup_start_lr=args.lr / 10,
                                   warmup_mode='linear')
    best_mae = 100
    best_rmse = None
    best_mape = None
    best_epoch = 0

    # write exp information
    log.info(f"==========Exp Args==========")
    log.info(f"directory:               {exp_dir}")
    log.info(f"name:                    {args.exp_name}")
    log.info(f"dataset:                 {args.dataset}")
    log.info(f"Use mult-size:           {args.multi_size_t}")
    log.info(f"Use lstm:                {args.lstm}")
    log.info(f"Use attention_snp:       {args.attention_snp}")
    log.info(f"Train_rate:              {args.train_rate}")
    log.info(f"val_test_data:           {args.val_test_data}")
    
    log.info(f"==========Model Args==========")
    log.info(f"heads:                   {args.heads}")
    log.info(f"pre_len:                 {args.pre_len}")
    log.info(f"seq_len:                 {args.seq_len}")
    log.info(f"batchsize:               {args.batchsize}")
    log.info(f"dropout:                 {args.dropout}")
    log.info(f"in_dim:                  {args.in_dim}")
    log.info(f"embed_size:              {args.embed_size}")
    log.info(f"warm_up:                 {args.warm_up}")

    log.info(f"==========Train Args==========")
    log.info(f"Epochs:                  {args.epochs}")
    log.info(f"Batchsize:               {args.batchsize}")
    log.info(f"lr:                      {args.lr}")
    log.info(f"weight_decay:            {args.weight_decay}")
    log.info(f"Model save dir:          {model_pkl_path}")

    train_len = len(train_dataloader)
    early_stop_cnt = 0
    for epoch in range(args.epochs):
        model.train()
        train_timer.epoch_start()
        total_loss = 0.0
        for batch, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            pre = model(x).reshape(-1, adj.shape[0])
            y = y.reshape(-1, adj.shape[0])

            loss = criterion(pre * std + mean, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warm_up:
                warmup_scheduler.step()
            total_loss += loss.item()

        model.eval()
        P = []
        L = []
        for x, y in val_dataloader:
            with torch.no_grad():
                x = x.to(device)
                pre = model(x) * std + mean
                P.append(pre.cpu().detach())
                L.append(y)
        pre = torch.cat(P, 0)
        label = torch.cat(L, 0)

        pre = pre.reshape(-1, adj.shape[0])
        label = label.reshape(-1, adj.shape[0])

        mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
        if epoch >= 0:
            if not os.path.exists(model_pkl_path):
                os.makedirs(model_pkl_path)
            model_batch_pkl_path = os.path.join(model_pkl_path, f"{epoch}_mae.pkl")
            torch.save(model.state_dict(), model_batch_pkl_path)
            log.info(f"model saved at {model_batch_pkl_path}")
        if mae < best_mae:
            early_stop_cnt = 0
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_pkl_path)
        else:
            early_stop_cnt += 1
        if epoch % 1 == 0:
            log.info(f"Current epoch: {epoch}\t MAE: {mae}\t RMSE: {rmse}\t MAPE: {mape}\t WAPE: {wape}")
            log.info(f"Best MAE at epoch: {best_epoch}\t MAE: {best_mae}\t RMSE: {best_rmse}\t MAPE: {best_mape} early_stop_cnt: {early_stop_cnt}")

        epoch_time , estimated_total_time, estimated_end_time = train_timer.epoch_end()
        log.info(f"Epoch {epoch} \tloss: {round(total_loss / train_len, 2)}\t lr: {round(lr_scheduler.get_lr()[0], 6)} \t epoch_time: {round(epoch_time,2)} \t estimated_total_time: {int(estimated_total_time)} \t estimated_end_time: {estimated_end_time}")
        if early_stop_cnt >= 20:
            print("early stop!")
            break


#     #----------------------------------------------- Test -----------------------------------------------
#     model.load_state_dict(torch.load(model_pkl_path))
#     model.eval()
#     P = []
#     L = []
#     for x, y in test_dataloader:
#         with torch.no_grad():
#             x = x.to(device)
#             pre = model(x) * std + mean
#             P.append(pre.cpu().detach())
#             L.append(y)
#     pre = torch.cat(P, 0)
#     label = torch.cat(L, 0)

#     pre = pre.reshape(-1, adj.shape[0])
#     label = label.reshape(-1, adj.shape[0])
    
#     mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
#     # mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())

#     log.info(f"=========={args.exp_name} Results==========")
#     log.info(f"Best MAE at epoch: {best_epoch}")
#     log.info(f"Eval MAE: {best_mae, }\t RMSE: {round(best_rmse,2)}\t MAPE: {best_mape}")
#     log.info(f"Test MAE: {mae}\t RMSE: {rmse}\t MAPE: {mape}\t WAPE: {wape}")
#     log.info("============================================")
