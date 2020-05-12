import torch
import torch.nn as nn
from torch.nn.functional import relu as r
from torch.nn.functional import softmax as smax
from datetime import datetime
import numpy as np

class VGG16(nn.Module):
    #https://cs.nju.edu.cn/wujx/paper/CNN.pdf
    def __init__(self):
        super(VGG16, self).__init__()
        #network
        self.conv1 = nn.Conv2d(3,64, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv2 = nn.Conv2d(64,64, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.pool1 = nn.MaxPool2d((2,2), stride=2)
        self.conv3 = nn.Conv2d(64, 128, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv4 = nn.Conv2d(128,128, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.pool2 = nn.MaxPool2d((2,2), stride=2)
        self.conv5 = nn.Conv2d(128,256, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv6 = nn.Conv2d(256,256, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv7 = nn.Conv2d(256,256, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.pool3 = nn.MaxPool2d((2,2), stride=2)
        self.conv8 = nn.Conv2d(256,512, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv9 = nn.Conv2d(512,512, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv10 = nn.conv2d(512,512, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.pool4 = nn.MaxPool2d((2,2), stride=2)
        self.conv11 = nn.conv2d(512,512, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv12 = nn.conv2d(512,512, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.conv13 = nn.conv2d(512,512, padding=1, kernel_size=(3,3), stride=1)
        #relu
        self.pool5 = nn.MaxPool2d((2,2), stride=2)
        self.linear1 = nn.Linear(7*7*512, 4096)
        #relu
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 1000)
        #softmax
        
        self.default_optimizer = torch.optim.SGD(VGG16.parameters(), lr=0.001, momentum=0.9)
        self.default_loss = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(r(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(r(x))
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(r(x))
        x = self.conv7(r(x))
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(r(x))
        x = self.conv10(r(x))
        x = self.pool4(x)
        x = self.conv11(x)
        x = self.conv12(r(x))
        x = self.conv13(r(x))
        x = self.pool5(x)
        x = self.linear1(x)
        x = self.dropout(r(x))
        x = self.linear(x)
        return smax(x)
    
    def fitDataPoint(self, x, y):
        #takes input and label, outputs loss object
        optimizer.zero_grad()
        loss = criteria(self.forward(x), y)
        optimizer.step()
        return loss
        
    def runValidation(self, x, y):
        # runs the validation test data, outputs loss object, no auto-grad
         with torch.no_grad():
            return criteria(self.forward(x), y)
        
    def save(self, fname = "VGG16-{0}".format(datetime.now().time().__str__())):
        torch.save(self.state_dict(), fname)
        print("Successfully Saved {0}".format(fname))
