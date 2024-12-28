import torch
from torchvision import datasets, transforms
import constants 

SEED = constants.SEED
cuda = constants.cuda

def load_mnist_data(batch_size=128):
    train_transforms = transforms.Compose([ transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                           transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                           transforms.ToTensor(), #this standardizes the data to be between 0 and 1
                                           transforms.Normalize((0.1307,), (0.3081,)) #this normalizes the data to have a mean of 0 and a standard deviation of 1
                                           ])
    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
    
    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transforms)
    # dataloader arguments 
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_loader, test_loader 

