""" 
STCVCNN模型
 """
import os
import sys
sys.path.append(os.getcwd())


import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexBatchNorm2d,ComplexMaxPool2d,ComplexConv2d,ComplexReLU,ComplexLinear,NaiveComplexBatchNorm2d

# input shape:[batchSize,Channel,Length,1]
class STCVCNN(nn.Module):
    def __init__(self,num_classes=3,runs=2) -> None:
        super(STCVCNN,self).__init__()
        self.num_classes=num_classes
        self.block1=nn.Sequential(
            ComplexConv2d(1,64,(3,1),padding=(1,0)),
            ComplexReLU(),
            ComplexMaxPool2d((2,1)),
            ComplexBatchNorm2d(64,track_running_stats = False),
        ) 
        self.block2=nn.Sequential(
            ComplexConv2d(64,64,(3,1),padding=(1,0)),
            ComplexReLU(),
            ComplexMaxPool2d((2,1)),
            ComplexBatchNorm2d(64),
        )
        self.block3=nn.Sequential(
            nn.Flatten(),
            ComplexLinear(576,1024),
            ComplexReLU(),
        )
        self.block4= ComplexLinear(1024,self.num_classes)

    def forward(self,x):
        x=self.block1(x)
        for i in range(8):
            x=self.block2(x)

        y_features=self.block3(x)
        y_classifier=self.block4(y_features)
        y_classifier = y_classifier.abs()
        y_classifier = F.softmax(y_classifier, dim=1)
        
        return y_features.abs(),y_classifier

if __name__=='__main__':
    a=torch.rand((4,1,4800,1),dtype=torch.complex64)
    print(a[0,0,:5,0])
    model=STCVCNN()
    # x=model(a)
    y_features,y_classifier=model(a)
    # print(x[0,0,:5,0])
    print(y_features.shape,y_classifier)