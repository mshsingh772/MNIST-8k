from data.load_data import load_mnist_data
from models.model3 import create_model as model3
from models.model2 import create_model as model2
from models.model1 import create_model as model1
from training.train import train_model,test_model
import constants
import torch.optim as optim
from torchsummary import summary
import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR
device = constants.device

def run_experiment(model_creator,EPOCHS=15):
    train_loader, test_loader = load_mnist_data()
    model = model_creator().to(device)
    
    print("Model Summary:")
    print(summary(model, input_size=(1, 28, 28)))
    # return True

    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_model(model, device, train_loader, optimizer)
        scheduler.step()
        test_model(model, device, test_loader)

def main():
    print("Running model training on device: ",device)
    run_experiment(model3)
    
    # print("\nRunning AnotherModel...")
    # run_experiment(create_another_model)

if __name__ == "__main__":
    main() 