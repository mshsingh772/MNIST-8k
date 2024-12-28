import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) 
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.conv5 = nn.Conv2d(256, 512, 3) 
        self.conv6 = nn.Conv2d(512, 1024, 3) 
        self.conv7 = nn.Conv2d(1024, 10, 3) 

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
def create_model():
    return Net() 

'''
Target:
    Get the set-up right
    Set Transforms
    Set Data Loader
    Set Basic Working Code
    Set Basic Training  & Test Loop
Results:
    Parameters: 6.3M
    Best Training Accuracy: 99.92
    Best Test Accuracy: 99.26
Analysis:
    Extremely Heavy Model, need less than 8k parameters.
    Model is over-fitting.
    Ignoring the initial epochs, the gap between training and test accuracy is too high.
'''