import torch
from torchvision import datasets, transforms

def calculate_mean_std(dataset):
    """Calculate the mean and standard deviation of a dataset."""
    # Create a DataLoader to iterate over the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize mean and standard deviation
    mean = 0.0
    std = 0.0
    total_samples = 0

    # Iterate over the dataset to calculate mean and std
    for data, _ in loader:
        batch_samples = data.size(0)  # Get the number of samples in the batch
        data = data.view(batch_samples, data.size(1), -1)  # Flatten the data
        mean += data.mean(2).sum(0)  # Sum the mean of each channel
        std += data.std(2).sum(0)  # Sum the std of each channel
        total_samples += batch_samples  # Update the total number of samples

    # Calculate the overall mean and std
    mean /= total_samples
    std /= total_samples

    return mean, std

def get_mnist_mean_std():
    """Get the mean and standard deviation for the MNIST dataset."""
    # Define a simple transform to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the MNIST training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Calculate and return the mean and std
    return calculate_mean_std(train_dataset) 

if __name__ == "__main__":
    # Calculate and print the mean and std for the MNIST dataset
    mean, std = get_mnist_mean_std()
    print(f"Mean: {mean}, Std: {std}")
