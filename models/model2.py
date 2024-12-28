import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False) 
        self.norm1 = nn.BatchNorm2d(8)
        self.drop = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1, bias=False) 
        self.norm2 = nn.BatchNorm2d(12)
        self.drop = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(12, 16, 3, padding=1, bias=False) 
        self.norm3 = nn.BatchNorm2d(16)
        self.drop = nn.Dropout2d(0.1)
        self.conv4 = nn.Conv2d(16, 20, 3, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(20)
        self.drop = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(2,2) 

        self.conv5 = nn.Conv2d(20, 24, 3, bias=False) 
        self.norm5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 28, 3, bias=False) 
        self.antman = nn.Conv2d(28, 10 , 1, bias=False)
        self.gap = nn.AvgPool2d(3)

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.norm1(x)
      x = self.drop(x)
      x = F.relu(self.conv2(x))
      x = self.norm2(x)
      x = self.drop(x)
      x = self.pool(x)

      x = F.relu(self.conv3(x))
      x = self.norm3(x)
      x = self.drop(x)
      x = F.relu(self.conv4(x))
      x = self.norm4(x)
      x = self.drop(x)
      x = self.pool(x)

      x = F.relu(self.conv5(x))
      x = self.norm5(x)
      x = F.relu(self.conv6(x))
      x = self.antman(x)
      x = self.gap(x)
      x = x.view(-1, 10)

      return F.log_softmax(x)

def create_model():
    return Net() 


'''
Target:
    Reduce number of parameters.
    Reduce gap between training and test.
    Use Batch Normalization and regularization.
    Use GAP will help reduce parameters.
Results:
    Parameters: 16,352
    Best Training Accuracy: 99.22
    Best Test Accuracy: 99.49
Analysis:
    Kept the architecture with two to three convolution blocks as the dataset is primarily just edges and gradients.
    The gap between training and test reduced by a lot.
    Reached the required test accuracy of 99.4% more than three times.
    Model is lighter however need more changes, need less than 8k parameters.
    Model is under-fitting, which makes sense as we have introduced regularization which makes training hard.
    Architecture is good, but need more changes.
'''