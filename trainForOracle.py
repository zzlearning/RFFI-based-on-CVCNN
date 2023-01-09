import argparse
import sys
import os
import importlib
import shutil
import time

import torch
import torch.nn as nn
from dataset.RFFIDataset import RFFIDataset
from torch.utils.data import DataLoader,Dataset
from network.ORACLE import ORACLE
from utils.strategy import step_lr, accuracy
from utils.plot import draw_curve
from utils.centerloss import CenterLoss
from utils.hard_triplet_loss import HardTripletLoss
from test import predict_unkonwn_sig

from sklearn.model_selection import train_test_split
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cfgs):
    trainDataset=RFFIDataset(txt_path=cfgs.train_txt_path,
                            norm_form=cfgs.norm_form,
                            offset=cfgs.offset,
                            bytes_of_one_sample=cfgs.bytes_of_one_sample
                            )
    validDataset=RFFIDataset(txt_path=cfgs.valid_txt_path,
                            norm_form=cfgs.norm_form,
                            offset=cfgs.offset,
                            bytes_of_one_sample=cfgs.bytes_of_one_sample
                            )

    train_loader=DataLoader( trainDataset,
                            batch_size=cfgs.batch_size,
                            num_workers=cfgs.num_workers,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True
                            )
    valid_loader=DataLoader( validDataset,
                            batch_size=cfgs.batch_size,
                            num_workers=cfgs.num_workers,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True
                            )    

    model = eval(f"{cfgs.model}")(num_classes=len(cfgs.classes), runs=len(cfgs.runs)).to(device, non_blocking=True)
    if cfgs.resume:
        model=torch.load(cfgs.resume_path)

    # log
    if not os.path.exists(f'./work_dir/{cfgs.checkpoint_name}'):
      os.makedirs(f'./work_dir/{cfgs.checkpoint_name}')
    with open(f'./work_dir/{cfgs.checkpoint_name}/Train_log.txt', 'a+') as f:
        f.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'-'*30+'\n')
        for name, value in cfgs.get_members().items():
            f.write(f"{name}: {value} \n")
        f.write("\n\n")

    criterion1 = nn.CrossEntropyLoss().to(device, non_blocking=True)  # 交叉熵损失
   
    # trian
    train_loss_sum = 0
    train_top1_sum = 0
    max_val_acc = 0
    train_draw_acc = []
    val_draw_acc = []
    train_draw_loss=[]
    val_draw_loss=[]
    lr = cfgs.lr

    for epoch in range(cfgs.epoches):
        ep_start = time.time()
        print('epoch:',epoch+1)
        top1_sum=0
        epoch_loss=0
        # 优化器
        # lr = step_lr(epoch, lr,T=20)
        optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=lr, betas=(0.9, 0.999), weight_decay=0)
        # optimizer=torch.optim.SGD(lr=cfgs.lr,params=model.parameters())
        model.train()
        sum = 0
        train_loss_sum = 0
        train_top1_sum = 0
        for i, (signal, label) in enumerate(train_loader):
            # signal是复数形式，且shape为（batch,1，128，1），将其转换为两通道，shape=（2，128）
            signal=torch.cat((signal.real[:,:,:,0],signal.imag[:,:,:,0]),dim=1)
            input = signal.to(device, non_blocking=True)
            target = label.to(device, non_blocking=True).long()
            y_classifier=model(input)     # 前向推理
            loss=criterion1(y_classifier,target)# 计算损失
            optimizer.zero_grad()# 优化器梯度清空
            loss.backward()# 反向传播
            optimizer.step() # 更新参数
            
            top1=accuracy(y_classifier.data,target.data,topk=(1,))# 计算top1分类准确率
            train_loss_sum += loss.item()
            train_top1_sum += top1[0]
            sum += 1
            top1_sum += top1[0]
            epoch_loss+=loss.item()

            if (i+1) % cfgs.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f'
                  %(epoch+1, cfgs.epoches, i+1, len(trainDataset)//cfgs.batch_size, 
                  lr, train_loss_sum/sum, train_top1_sum/sum))
                with open(f'./work_dir/{cfgs.checkpoint_name}/Train_log.txt', 'a+') as f:
                    f.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f\n'
                        %(epoch+1, cfgs.epoches, i+1, len(trainDataset)//cfgs.batch_size, 
                        lr, train_loss_sum/sum, train_top1_sum/sum))
                sum = 0
                train_loss_sum = 0
                train_top1_sum = 0

        train_draw_acc.append(top1_sum.item()/len(train_loader))
        train_draw_loss.append(epoch_loss/len(train_loader))
        
        epoch_time = (time.time() - ep_start) / 60.
        
        # valid
        if epoch % cfgs.valid_freq == 0 and epoch < cfgs.epoches:
            val_time_start = time.time()
            sum=0
            val_loss_sum = 0
            val_top1_sum = 0
            model.eval()
            with torch.no_grad():
                for i,(signal, label) in enumerate(valid_loader):
                    signal=torch.cat((signal.real[:,:,:,0],signal.imag[:,:,:,0]),dim=1)
                    input_val = signal.to(device, non_blocking=True)
                    target_val = label.to(device, non_blocking=True).long()
                    val_y_classifier = model(input_val)
                    loss=criterion1(val_y_classifier,target_val)# 计算损失
                    top1_val = accuracy(val_y_classifier.data, target_val.data, topk=(1,))
                    sum+=1
                    val_loss_sum += loss.data.cpu().numpy()
                    val_top1_sum += top1_val[0]
            val_loss = val_loss_sum / sum
            val_top1 = val_top1_sum / sum
            val_draw_acc.append(val_top1.item())
            val_draw_loss.append(val_loss.item())
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f'
                    %(epoch+1, cfgs.epoches, val_loss, val_top1, val_time*60, max_val_acc))
            print('epoch time: {}s'.format(epoch_time*60))
            print('valid time: {}s'.format(val_time*60))

            if val_top1[0].data > max_val_acc:
                max_val_acc = val_top1[0].data
                print('Taking snapshot...')
                torch.save(model, f'./work_dir/{cfgs.checkpoint_name}/{cfgs.checkpoint_name}.pth')
            with open(f'./work_dir/{cfgs.checkpoint_name}/Train_log.txt', 'a+') as f:
                    f.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, val_time: %.4f s, max_val_acc: %4f\n'
                  %(epoch+1, cfgs.epoches, val_loss, val_top1, val_time*60, max_val_acc))
    draw_curve(train_draw_acc, val_draw_acc, f'./work_dir/{cfgs.checkpoint_name}/Train_Acc_{int(max_val_acc)}.png',acc_or_loss='acc')
    draw_curve(train_draw_loss,val_draw_loss,f'./work_dir/{cfgs.checkpoint_name}/Train_Loss.png',acc_or_loss='loss')
    with open(f'./work_dir/{cfgs.checkpoint_name}/Train_log.txt', 'a+') as f:
        f.write('-'*40+"End of Train"+'-'*40+'\n')



if __name__=='__main__':
    parser = argparse.ArgumentParser(description=('Train a model for RFFI dataset.'))
    parser.add_argument(
        '--config', 
        default=r'configs/configs02.py',
        type=str,
        help='Configuration file Path'
    )
    args = parser.parse_args()

    # 动态加载配置文件
    sys.path.append(os.path.dirname(args.config))
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    cfgs = importlib.import_module(module_name).Configs()

    # 保存配置
    if not os.path.exists(f'./work_dir/{cfgs.checkpoint_name}'):
        os.makedirs(f'./work_dir/{cfgs.checkpoint_name}')
    shutil.copy(args.config, f'./work_dir/{cfgs.checkpoint_name}/Train_configs.py')

    # GPU指定
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.gup_id
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练 & 验证
    train(cfgs)

    # 测试
    # predict_unkonwn_sig(cfgs)