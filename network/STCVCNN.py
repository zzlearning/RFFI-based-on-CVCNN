""" 
STCVCNN模型
 """

import torch
import torch.nn as nn


class STCVCNN(nn.Module):
    def __init__(self,num_classes,runs) -> None:
        super(STCVCNN,self).__init__()
        self.num_classes=num_classes
        self.block=nn.Conv1d(1000,num_classes,dtype=torch.complex128)

    def forward(self,x):
        x=self.block(x)
        return x
