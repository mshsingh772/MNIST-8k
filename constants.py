import torch

SEED = 1
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")