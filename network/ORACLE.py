""" 
ORACLE模型
 """
import os
import sys
sys.path.append(os.getcwd())


import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d,ComplexMaxPool2d,ComplexConv2d,ComplexReLU,ComplexLinear,NaiveComplexBatchNorm2d

# input shape:[batchSize,Channel,Length,1]
class ORACLE(nn.Module):
    def __init__(self,num_classes=3,runs=2) -> None:
        super(ORACLE,self).__init__()
        self.num_classes=num_classes
        self.block1=nn.Sequential(
            nn.Conv1d(2,128,7,padding=3),
            nn.ReLU(),
            nn.Conv1d(128,128,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        ) 
        self.block2=nn.Sequential(
            nn.Conv1d(128,128,7,padding=3),
            nn.ReLU(),
            nn.Conv1d(128,128,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        ) 
        self.block3=nn.Sequential(
            nn.Conv1d(128,128,7,padding=3),
            nn.ReLU(),
            nn.Conv1d(128,128,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        ) 
        self.block4=nn.Sequential(
            nn.Conv1d(128,128,7,padding=3),
            nn.ReLU(),
            nn.Conv1d(128,128,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        ) 
        self.fc1=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128*8,256),
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU()
        )
        self.softmax=nn.Sequential(
            nn.Linear(128,self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.softmax(x)
        return x

if __name__=='__main__':
    a=torch.rand((4,2,128))
    print(a[0,0,:5])
    model=ORACLE()
    # x=model(a)
    y=model(a)
    # print(x[0,0,:5,0])
    print(y.shape,y)