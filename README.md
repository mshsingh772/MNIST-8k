# MNIST-8k
This project contains multiple neural network models for training on the MNIST dataset. Each model is designed with specific targets, and the results and analysis are documented to understand the performance and improvements.

## Project Structure

- **data/**: Contains data loading scripts.
- **models/**: Contains different model architectures.
- **training/**: Contains training and testing scripts.
- **utils.py**: Utility functions for data preprocessing.
- **run_experiments.py**: Script to run experiments with different models.
- **main.py**: Entry point for running experiments.

## Models

### Model 1

**Target:**
- Get the setup right.
- Set transforms, data loader, and basic working code.
- Establish basic training and test loop.

**Results:**
- Parameters: 6.3M
- Best Training Accuracy: 99.92%
- Best Test Accuracy: 99.26%

**Analysis:**
- Extremely heavy model, need less than 8k parameters.
- Model is overfitting.
- Ignoring the initial epochs, the gap between training and test accuracy is too high.

### Model 2

**Target:**
- Reduce the number of parameters.
- Reduce the gap between training and test accuracy.
- Use batch normalization and regularization.
- Use GAP to help reduce parameters.

**Results:**
- Parameters: 16,352
- Best Training Accuracy: 99.22%
- Best Test Accuracy: 99.49%

**Analysis:**
- Architecture with two to three convolution blocks as the dataset is primarily just edges and gradients.
- The gap between training and test reduced significantly.
- Reached the required test accuracy of 99.4% more than three times.
- Model is lighter but needs more changes, aiming for less than 8k parameters.
- Model is underfitting due to regularization, making training hard.
- Architecture is good but needs more changes.

### Model 3

**Target:**
- Reduce the number of parameters.
- Maintain accuracy of 99.4% more than 2-3 times.

**Results:**
- Parameters: 7,944
- Best Training Accuracy: 99.25
- Best Test Accuracy: 99.46

**Analysis:**
- Introduced step LR scheduler.
- Included more augmentation techniques.
- Tinkering with LR values helped reach 99.4% accuracy in last 7 epochs.

## Logs  
- Model Summary:  
----------------------------------------------------------------  
        Layer (type)               Output Shape         Param #  
----------------------------------------------------------------
            Conv2d-1           [-1, 10, 26, 26]              90  
       BatchNorm2d-2           [-1, 10, 26, 26]              20  
              ReLU-3           [-1, 10, 26, 26]               0  
           Dropout-4           [-1, 10, 26, 26]               0  
            Conv2d-5           [-1, 15, 24, 24]           1,350  
       BatchNorm2d-6           [-1, 15, 24, 24]              30  
              ReLU-7           [-1, 15, 24, 24]               0  
           Dropout-8           [-1, 15, 24, 24]               0  
            Conv2d-9           [-1, 10, 24, 24]             150  
        MaxPool2d-10           [-1, 10, 12, 12]               0  
           Conv2d-11           [-1, 16, 10, 10]           1,440  
      BatchNorm2d-12           [-1, 16, 10, 10]              32  
             ReLU-13           [-1, 16, 10, 10]               0  
          Dropout-14           [-1, 16, 10, 10]               0  
           Conv2d-15             [-1, 16, 8, 8]           2,304  
      BatchNorm2d-16             [-1, 16, 8, 8]              32  
             ReLU-17             [-1, 16, 8, 8]               0  
          Dropout-18             [-1, 16, 8, 8]               0  
           Conv2d-19             [-1, 16, 6, 6]           2,304  
      BatchNorm2d-20             [-1, 16, 6, 6]              32  
             ReLU-21             [-1, 16, 6, 6]               0  
          Dropout-22             [-1, 16, 6, 6]               0  
        AvgPool2d-23             [-1, 16, 1, 1]               0  
           Conv2d-24             [-1, 10, 1, 1]             160  
================================================================  
Total params: 7,944  
Trainable params: 7,944  
Non-trainable params: 0  
----------------------------------------------------------------  
Input size (MB): 0.00  
Forward/backward pass size (MB): 0.62  
Params size (MB): 0.03  
Estimated Total Size (MB): 0.66  
----------------------------------------------------------------  
None  
EPOCH: 0  
Loss=0.0810 Batch_id=468 Accuracy=93.30: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 54.50it/s]  

Test set: Average loss: 0.0769, Accuracy: 9750/10000 (97.50%)  

EPOCH: 1  
Loss=0.2008 Batch_id=468 Accuracy=97.63: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.61it/s]  

Test set: Average loss: 0.0458, Accuracy: 9854/10000 (98.54%)  

EPOCH: 2  
Loss=0.0400 Batch_id=468 Accuracy=98.20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.35it/s]  

Test set: Average loss: 0.0376, Accuracy: 9874/10000 (98.74%)  

EPOCH: 3  
Loss=0.0108 Batch_id=468 Accuracy=98.44: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 54.92it/s]  

Test set: Average loss: 0.0363, Accuracy: 9881/10000 (98.81%)  

EPOCH: 4  
Loss=0.0309 Batch_id=468 Accuracy=98.40: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.37it/s]  

Test set: Average loss: 0.0282, Accuracy: 9908/10000 (99.08%)  

EPOCH: 5  
Loss=0.0359 Batch_id=468 Accuracy=98.66: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.71it/s]  

Test set: Average loss: 0.0392, Accuracy: 9876/10000 (98.76%)  

EPOCH: 6  
Loss=0.1507 Batch_id=468 Accuracy=99.03: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.46it/s]  

Test set: Average loss: 0.0205, Accuracy: 9935/10000 (99.35%)  

EPOCH: 7  
Loss=0.0041 Batch_id=468 Accuracy=99.12: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.19it/s]  

Test set: Average loss: 0.0188, Accuracy: 9943/10000 (99.43%)  

EPOCH: 8  
Loss=0.0728 Batch_id=468 Accuracy=99.17: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 56.31it/s]  

Test set: Average loss: 0.0184, Accuracy: 9944/10000 (99.44%)  

EPOCH: 9  
Loss=0.0281 Batch_id=468 Accuracy=99.19: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.51it/s]  

Test set: Average loss: 0.0179, Accuracy: 9941/10000 (99.41%)  

EPOCH: 10  
Loss=0.0039 Batch_id=468 Accuracy=99.22: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.62it/s]  

Test set: Average loss: 0.0191, Accuracy: 9940/10000 (99.40%)  

EPOCH: 11  
Loss=0.0105 Batch_id=468 Accuracy=99.21: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.29it/s]  

Test set: Average loss: 0.0177, Accuracy: 9946/10000 (99.46%)  

EPOCH: 12  
Loss=0.0121 Batch_id=468 Accuracy=99.26: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.91it/s]  

Test set: Average loss: 0.0184, Accuracy: 9940/10000 (99.40%)  

EPOCH: 13  
Loss=0.0137 Batch_id=468 Accuracy=99.22: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 54.87it/s]  

Test set: Average loss: 0.0178, Accuracy: 9940/10000 (99.40%)  

EPOCH: 14  
Loss=0.0065 Batch_id=468 Accuracy=99.25: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 469/469 [00:08<00:00, 55.85it/s]  

Test set: Average loss: 0.0181, Accuracy: 9941/10000 (99.41%)


## Usage

1. **Install Dependencies**: Ensure you have PyTorch and torchvision installed.
2. **Run Experiments**: Use `main.py` to run experiments with different models.
   ```bash
   python main.py
   ```

## Conclusion

This project demonstrates the iterative process of model development, focusing on reducing parameters while maintaining high accuracy. Each model builds upon the previous one, incorporating new techniques and optimizations.