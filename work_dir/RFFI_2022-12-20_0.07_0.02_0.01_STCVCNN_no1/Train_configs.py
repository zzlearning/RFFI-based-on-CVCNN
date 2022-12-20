import time
import os
TIME=time.strftime('%Y-%m-%d', time.localtime())

class Configs(object):
    def __init__(self):

        self.note=''                                        # 实验备注

        """  数据集划分参数配置 """
        self.data_path=r'D:\RFFI-based-on-CVCNN\datasets\ORACLE\neu_m044q5210\KRI-16Devices-RawData\2ft'               # 要划分的数据路径
        self.save_path='./dataset/'                         # 划分好的索引文件存放目录
        self.trainset_ratio=0.07                             # 训练数据比例
        self.validset_ratio=0.02                             # 验证数据比例
        self.testset_ratio=0.01                              # 测试数据比例
        self.dataset_name=f'RFFI_{TIME}_{self.trainset_ratio}_{self.validset_ratio}_{self.testset_ratio}'    # 划分好的数据集索引文件名字
        self.sig_len= 1000                                  # 信号裁剪长度
        self.classes=['3123D7B','3123D7D','3123D7E']                    # 信号类别
        self.runs=['run1']                           # 信号采集的序号
        self.continuous=False                               # 是否进行连续划分

        self.oneFilePath=r"D:\RFFI-based-on-CVCNN\datasets\ORACLE\neu_m044q5210\KRI-16Devices-RawData\2ft\WiFi_air_X310_3123D7B_2ft_run1.sigmf-data"


        """ Dataset """
        dir_path=r'dataset\RFFI_2022-12-20_0.07_0.02_0.01'
        self.train_txt_path=os.path.join(dir_path,'train_set.txt')
        self.valid_txt_path=os.path.join(dir_path,'valid_set.txt')
        self.test_txt_path=os.path.join(dir_path,'test_set.txt')

        """ 训练参数 """
        # model
        self.model='STCVCNN'                            #实验模型
        self.exp_num = 1                                # 实验次数
        self.note = "no1"                                    # 实验备注
        self.gup_id = "0"                               # 指定GPU
        self.resume = False                               # 是否加载训练好的模型
        self.resume_path = "" # 加载模型路径

        # train
        self.epoches=2
        self.lr=0.001

        # log
        self.checkpoint_name=self.dataset_name+'_'+self.model+'_'+self.note

    def get_members(self):
        return vars(self)