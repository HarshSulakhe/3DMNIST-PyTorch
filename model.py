import torch
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel,self).__init__()
        self.conv1 = nn.Conv3d(3,8,3)
        self.conv2 = nn.Conv3d(8,32,3)
        self.conv3 = nn.Conv3d(32,32,3)
        # self.conv4 = nn.Conv3d(64,128,3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool3d(4)
        # self.pool2 = nn.MaxPool3d(2)
        # self.linear1 = nn.Linear(128*8,128)
        self.linear2 = nn.Linear(32,10)
        self.batchnorm = nn.BatchNorm3d(32)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.pool1(x)
        x = self.batchnorm(x)
        x = x.view(x.size()[0],-1)
        # x = self.linear1(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel,self).__init__()
        self.l1 = nn.Linear(4096,1024)
        self.l2 = nn.Linear(1024,256)
        self.l3 = nn.Linear(256,64)
        self.l4 = nn.Linear(64,10)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = x.view(-1,4096)
        x = self.l1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.elu(x)
        x = self.l4(x)
        return x
