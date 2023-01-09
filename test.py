import argparse
import sys
import os
import importlib
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
from dataset.RFFIDataset import RFFIDataset
from torch.utils.data import DataLoader,Dataset
from network.STCVCNN import STCVCNN
from utils.strategy import step_lr, accuracy
from utils.plot import draw_curve, plot_confusion_matrix

from sklearn.model_selection import train_test_split


# def TestDataset(num):
#     x = np.load(f"D:/RFFI-based-on-CVCNN/codes/FS-SEI/FS-SEI_4800/Dataset/X_test_{num}Class.npy")
#     y = np.load(f"D:/RFFI-based-on-CVCNN/codes/FS-SEI/FS-SEI_4800/Dataset/Y_test_{num}Class.npy")

#     x=(x-x.min())/(x.max()-x.min())
#     x=x[:,:,0]+x[:,:,1]*1j
#     x=x.astype(np.complex64)
#     x=np.expand_dims(x,axis=(1,3))
#     # print(type(x[0,0,0,0]))
#     y = y.astype(np.uint8)# 8bit数0-255
#     # X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.1, random_state= 30)
#     return x,y

# class FewShotDataset(Dataset):
#     def __init__(self,datas,labels):
#         self.datas=datas
#         self.labels=labels

#     def __getitem__(self, index):
#         data=self.datas[index]
#         label=self.labels[index]
#         return data ,label

#     def __len__(self):
#         return len(self.labels)

def predict_unkonwn_sig(cfgs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # X_test,Y_test=TestDataset(10)
    # testDataset=FewShotDataset(X_test,Y_test)

    testDataset=RFFIDataset(txt_path=cfgs.test_txt_path,
                            norm_form=cfgs.norm_form,
                            offset=cfgs.offset,
                            bytes_of_one_sample=cfgs.bytes_of_one_sample
                            )
   
    test_loader=DataLoader( testDataset,
                            batch_size=cfgs.batch_size,
                            num_workers=cfgs.num_workers,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True
                            )

    # model   
    model = torch.load(f'./work_dir/{cfgs.checkpoint_name}/{cfgs.checkpoint_name}.pth') 
    model=model.to(device, non_blocking=True)


    # log
    if not os.path.exists(f'./work_dir/{cfgs.checkpoint_name}'):
        os.makedirs(f'./work_dir/{cfgs.checkpoint_name}')
    with open(f'./work_dir/{cfgs.checkpoint_name}/Test_log.txt', 'a+') as f:
        f.write('-'*30+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'-'*30+'\n')
        for name, value in cfgs.get_members().items():
            f.write(f"{name}: {value} \n")
        f.write("\n\n")
    
    # start test
    val_time_start = time.time()
    model.eval()
    preds_list=[]
    labels_list=[]
    for i, (signal, label) in enumerate(test_loader):
        input_val = signal.to(device, non_blocking=True)
        y_features,y_classifier = model(input_val)
        
        _, preds = y_classifier.topk(1, 1, True, True)
        preds = preds.t()[0].detach().cpu().numpy()
        
        # save
        preds_list += preds.tolist()
        labels_list += label.tolist()
    val_time = (time.time() - val_time_start) / 60.
    accuracy = np.sum(np.array(preds_list)==np.array(labels_list))/len(labels_list)
    print("acc:", accuracy)
    
    plot_confusion_matrix(labels_list, preds_list, cfgs.classes, f'./work_dir/{cfgs.checkpoint_name}/Test_Confusion_Matrix_{int(accuracy*100)}.png', title=str(accuracy))
    with open(f'./work_dir/{cfgs.checkpoint_name}/Test_log.txt', 'a+') as f:
        f.write("\nACC: "+str(accuracy)+"\n")
        f.write(f"val_time: {val_time}\n")
        f.write('-'*40+f"End of test"+'-'*40+'\n')



if __name__=='__main__':
    parser = argparse.ArgumentParser(description=('Train a model for RFFI dataset.'))
    parser.add_argument(
        '--config', 
        default=r'configs/configs03.py',
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

    # 测试
    predict_unkonwn_sig(cfgs)