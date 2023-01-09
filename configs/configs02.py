import time
import os

NOTE='IQImbalance_c16_more'     # 数据集划分备注

class Configs(object):
    def __init__(self):

        self.note=''                                        # 实验备注

        """  数据集划分参数配置 """
        self.data_path=r'D:\RFFI-based-on-CVCNN\datasets\ORACLE\neu_m044q523j\KRI-16IQImbalances-DemodulatedData'               # 要划分的数据路径
        self.save_path='./dataset/'                         # 划分好的索引文件存放目录
        self.trainset_ratio=0.7                             # 训练数据比例
        self.validset_ratio=0.2                             # 验证数据比例
        self.testset_ratio=0.1                              # 测试数据比例
        self.dataset_name=f'RFFI_{NOTE}_{self.trainset_ratio}_{self.validset_ratio}_{self.testset_ratio}'    # 划分好的数据集索引文件名字
        self.bytes_of_one_sample=16                              # 一个信号采样点占用字节数
        self.offset=0                                       # 信号文件头部偏移量
        self.sig_len= 128                                  # 信号裁剪长度
        # self.classes=['3123D52', '3123D54', '3123D58', '3123D64', '3123D65', '3123D70', '3123D76', \
        # '3123D78', '3123D79', '3123D7B', '3123D7D', '3123D7E', '3123D80', '3123D89', '3123EFE', '3124E4A']                    # 信号类别
        self.classes=['IQ#1','IQ#2','IQ#3','IQ#4','IQ#5','IQ#7','IQ#9','IQ#13','IQ#14','IQ#15','IQ#17','IQ#18','IQ#19','IQ#25','IQ#26','IQ#32']
        self.runs=['run1']                           # 信号采集的序号
        self.continuous=True                             # 是否进行连续划分
        self.sliding_window=0.2                        # 滑动窗口的步长占比


        """ Dataset """
        dataset_path=self.dataset_name          # 默认情况
        self.train_txt_path=os.path.join('dataset',dataset_path,'train_set.txt')
        self.valid_txt_path=os.path.join('dataset',dataset_path,'valid_set.txt')
        self.test_txt_path=os.path.join('dataset',dataset_path,'test_set.txt')
        self.norm_form=None               # 归一化方式，None,'maxmin','z-score'

        """ 训练参数 """
        # model
        self.model='ORACLE'                            #实验模型
        self.exp_num = 1                                # 实验次数
        self.note = "triplet"                                    # 实验备注
        self.gup_id = "1"                               # 指定GPU
        self.resume = False                              # 是否加载训练好的模型
        self.resume_path = r"D:\RFFI-based-on-CVCNN\codes\RFFI-based-on-CVCNN\work_dir\RFFI_IQunbalance_c16_0.7_0.2_0.1_ORACLE_triplet\RFFI_IQunbalance_c16_0.7_0.2_0.1_ORACLE_triplet.pth" # 加载模型路径

        # train
        self.batch_size=1024
        self.epoches=20
        self.lr=0.0001
        self.num_workers = 2
        self.valid_freq=1

        # log
        self.iter_smooth=10           # 
        self.checkpoint_name=dataset_path+'_'+self.model+'_'+self.note

    def get_members(self):
        return vars(self)