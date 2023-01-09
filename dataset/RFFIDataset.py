# 张卓 创建于2022/12/4
# 对数据文件进行划分和加载，提供给模型使用

import os
import random
import numpy as np
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader


import argparse
import sys
import importlib

class RFFIDataset(Dataset):
    def __init__(self,txt_path,norm_form=None,offset=0,bytes_of_one_sample=16):
        self.labels,self.sigs_path,self.start_poses,self.sigs_len=parse_txt(txt_path)
        self.norm_form=norm_form
        self.bytes_of_one_sample=bytes_of_one_sample
        self.offset=offset
    def __getitem__(self,index):
        label=self.labels[index]
        sig=get_sig(self.sigs_path[index],self.start_poses[index],self.sigs_len[index],
                    offset=self.offset,
                    bytes_of_one_sample=self.bytes_of_one_sample)

        # 需要归一化/标准化
        if self.norm_form is not None:
            sig=sig_norm(sig,self.norm_form)

        
        return sig,label

    def __len__(self):
        return len(self.labels)
    
def parse_txt(path):
    """ 
    Function:
        解析数据集索引文件，以列表形式返回信号标签、信号绝对路径、信号开始位置、信号采样长度。
    Arguments:
        path: str, 数据集索引文件路径
    Returns:
        labels: list, 信号标签
        sig_path: list, 信号绝对路径
        start_pos: list, 信号开始位置
        sig_len: list, 信号采样长度
        """
    labels,sigs_path,start_poses,sigs_len=[],[],[],[]
    with open(path,'r') as f:
        for line in f.readlines():
            label,sig_path,start_pos,sig_len=line.strip().split(',')
            labels.append(int(label))
            sigs_path.append(sig_path)
            start_poses.append(int(start_pos))
            sigs_len.append(int(sig_len))
    return labels, sigs_path, start_poses, sigs_len

def get_sigmf_data_len(sig_path, offset = 0, bytes_of_one_sample=16):
    """ 
    Function:
        获取.sigmf-data文件的采样点个数
    Arguments:
        sig_path: str, 信号路径 
        offset: int, sigmf-data文件头部信息长度
        bytes_of_one_sample: int, 一个采样点占用字节数目
    Returns:
        dataLen: int, sigmf-data文件所含信号采样点数
    """
    with open(sig_path, 'rb') as f_sig:
        f_sig.seek(0,2)            #指针移至末尾
        # 计算可读采样点数
        sampleLen = (f_sig.tell()-offset)//bytes_of_one_sample
        return sampleLen

def get_sig(sig_path,start_pos=0,sig_len=100,offset=0,bytes_of_one_sample=16):
    """ 
    Function:
        从sig_path中获取一段信号,返回一行复数形式的信号
    Arguments:
        sig_path: str, 信号绝对路径
        start_pos: int, 信号开始位置
        sig_len: int, 信号长度
    Return:
        sig: np.ndrray,dtype=np.complex64, 信号序列
    """
    # 判断采样序列是否溢出
    total_sample_point_capacity=get_sigmf_data_len(sig_path,offset,bytes_of_one_sample)
    if start_pos+sig_len>total_sample_point_capacity:
        print(start_pos,sig_len,total_sample_point_capacity,sig_path)
        raise Exception('序列划分溢出')

    with open(sig_path,'rb') as f:
        f.seek(offset+start_pos*bytes_of_one_sample,0)
        byte_stream = f.read(bytes_of_one_sample*sig_len) #成对(双通道)读取数据
        sig=np.frombuffer(byte_stream,dtype=np.complex128,count=-1,offset=0).astype(np.complex64).reshape(1,-1,1)
    return sig      # shape=(sig_len,)

def sig_norm(sig,norm_form='maxmin'):
    """ 
    Function:
        将复数信号sig进行归一化/标准化处理
    Arguments:
        sig: ndarry, 复数信号
        norm_form: str, 归一化方式,可选maxmin、z-score
    Return:
        sig: np.ndrray,dtype=np.complex64, 信号序列
     """
    if norm_form not in ['maxmin','z-score']:
        raise Exception('标准化出错')
    eps=1e-5
    if norm_form=='maxmin':
        real_min=sig.real.min()
        real_max=sig.real.max()
        sig.real=(sig.real-real_min)/(real_max-real_min+eps)

        imag_min=sig.imag.min()
        imag_max=sig.imag.max()
        sig.imag=(sig.imag-imag_min)/(imag_max-imag_min+eps)

        return sig
    elif norm_form=='z-score':
        real_mean=sig.real.mean()
        real_std=sig.real.std()
        sig.real=(sig.real-real_mean)/(real_std+eps)

        imag_mean=sig.imag.mean()
        imag_std=sig.imag.std()
        sig.imag=(sig.imag-imag_mean)/(imag_std+eps)

        return sig
        

def devide_dataset( data_path,
                    save_path='./dataset',
                    dataset_name='ORACLE_devide_data',
                    sig_len=10000,
                    classes=['3723D7B','3723D7D','3723D7E'],
                    runs=['run1'],
                    continuous=False,
                    trainset_ratio=0.5,
                    validset_ratio=0.25,
                    testset_ratio=0.25,
                    bytes_of_one_sample=16,
                    offset=0,
                    sliding_window=0):
    """ 
    Function:
        划分数据集,分为训练集,验证集,和测试集,以索引形式保存在txt文件中,索引包含数据标签,数据路径,开始位置,采样长度
        数据的存放格式应为:data_path:
                            *classes[0]_runs[0].sigmf-data
                            ...
    Arguments:
        data_path: str, 要划分的数据路径
        save_path: str, 划分好的索引文件存放目录
        dataset_name: str, 划分好的数据集索引文件名字
        sig_len: int, 信号裁剪长度
        classes: list, 信号类别
        runs: list, 信号采集的序号
        continuous: bool, 是否进行连续划分
        trainset_ratio: float, 训练数据比例
        validset_ratio: float, 验证数据比例
        testset_ratio: float,测试数据比例
    Return:
        没有返回值,生成txt文件保存在本地:save_path/dataset_name/*.txt
        """
    save_dir=str(f"{save_path}/{dataset_name}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if trainset_ratio != 0:
        train_txt=open(f'{save_path}/{dataset_name}/train_set.txt', 'w')
    if validset_ratio != 0:
        valid_txt = open(f'{save_path}/{dataset_name}/valid_set.txt', 'w')
    if testset_ratio != 0:
        test_txt = open(f'{save_path}/{dataset_name}/test_set.txt', 'w')
    
    for file in os.listdir(data_path):
        # file name示例:WiFi_air_X310_3123D7B_2ft_run1.sigmf-data
        extension=os.path.splitext(file)[1]
        file_class=os.path.splitext(file)[0].split('_')[5]
        file_run=os.path.splitext(file)[0].split('_')[-1]
        if file_class in classes:
            label=np.where(np.array(classes)==file_class)[0][0]
        else:
            continue
        if (extension=='.sigmf-data')&(file_class in classes)&(file_run in runs):
            sigmf_path=os.path.join(data_path,file)
            crop_num=get_sigmf_data_len(sigmf_path,offset,bytes_of_one_sample)//sig_len-1

            if not continuous:
                # 训练集
                train_idx = random.sample(range(crop_num), int(crop_num*trainset_ratio))
                residue_idx = list(set(range(crop_num)).difference(set(train_idx)))
                # 验证集
                valid_idx=random.sample(residue_idx,int(crop_num*validset_ratio))
                residue_idx = list(set(range(crop_num)).difference(set(train_idx)).difference(valid_idx))
                # 测试集
                test_idx=random.sample(residue_idx,int(crop_num*testset_ratio))

            if continuous:
                train_idx=[i for i in range(int(crop_num*trainset_ratio))]
                valid_idx=[i for i in range(int(crop_num*trainset_ratio),int(crop_num*(trainset_ratio+validset_ratio)))]
                test_idx=[i for i in range(int(crop_num*(trainset_ratio+validset_ratio)),int(crop_num*(trainset_ratio+validset_ratio+testset_ratio)))]
                random.shuffle(train_idx)
                random.shuffle(valid_idx)
                random.shuffle(test_idx)

            # 保存到txt
            if sliding_window!=0:
                for idx in train_idx:
                    for window in range(int(1/sliding_window)):
                        train_txt.write(f"{label},{sigmf_path},{idx*sig_len+int(sig_len*sliding_window*window)},{sig_len}\n")
                for idx in valid_idx:
                    for window in range(int(1/sliding_window)):
                        valid_txt.write(f"{label},{sigmf_path},{idx*sig_len+int(sig_len*sliding_window*window)},{sig_len}\n")
                for idx in test_idx:
                    for window in range(int(1/sliding_window)):
                        test_txt.write(f"{label},{sigmf_path},{idx*sig_len+int(sig_len*sliding_window*window)},{sig_len}\n")

            else:
                for idx in train_idx:
                    train_txt.write(f"{label},{sigmf_path},{idx*sig_len},{sig_len}\n")
                for idx in valid_idx:
                    valid_txt.write(f"{label},{sigmf_path},{idx*sig_len},{sig_len}\n")
                for idx in test_idx:
                    test_txt.write(f"{label},{sigmf_path},{idx*sig_len},{sig_len}\n")

# 划分数据集
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Divide/test the dataset'
    )
    parser.add_argument(
        '--config', 
        default=r'./configs/configs02.py',
        type=str,
        help='Configuration file Path'
    )
    args = parser.parse_args()

    # 动态加载配置文件
    sys.path.append(os.path.dirname(args.config))
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    cfgs = importlib.import_module(module_name).Configs()

    # sig=get_sig(sig_path=cfgs.oneFilePath,start_pos=0,sig_len=20)
    # print(sig)
    # 划分数据集
    # devide_dataset( 
    #     data_path=cfgs.data_path,
    #     save_path=cfgs.save_path,
    #     dataset_name=cfgs.dataset_name,
    #     sig_len=cfgs.sig_len,
    #     classes=cfgs.classes,
    #     runs=cfgs.runs,
    #     continuous=cfgs.continuous,
    #     trainset_ratio=cfgs.trainset_ratio,
    #     validset_ratio=cfgs.validset_ratio,
    #     testset_ratio= cfgs.testset_ratio,
    #     bytes_of_one_sample=cfgs.bytes_of_one_sample,
    #     offset=cfgs.offset,
    #     sliding_window=cfgs.sliding_window
    # )

    trainDataset=RFFIDataset(txt_path=cfgs.train_txt_path)
    # validDataset=RFFIDataset(txt_path=cfgs.valid_txt_path)
    # testDataset=RFFIDataset(txt_path=cfgs.test_txt_path)


    train_loader=DataLoader( trainDataset,
                            batch_size=4,
                            num_workers=0,
                            shuffle=True,
                            drop_last=False
                            )
    # valid_loader=DataLoader( validDataset,
    #                         batch_size=4,
    #                         num_workers=0,
    #                         shuffle=True,
    #                         drop_last=False
    #                         )                        
    # test_loader=DataLoader( testDataset,
    #                         batch_size=4,
    #                         num_workers=0,
    #                         shuffle=True,
    #                         drop_last=False
    #                         )
    import matplotlib.pyplot as plt
    for (data,label),i in zip(train_loader,range(8)):
        print(data[0,0,:10,0],label)
        # plt.scatter(data.real[0,0,:,0],data.imag[0,0,:,0])
        # plt.title(label[0])
        # plt.show()