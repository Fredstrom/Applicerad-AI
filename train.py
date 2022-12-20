from model_parameters import settings
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from model import CNN_model
import torch
from datamanager import DataManager
from glob import glob


def init_model():
    dm = DataManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are using: {device}")

    train_loader = dm.train_loader
    val_loader = dm.val_loader
    print(f"Size of training set: {len(glob(dm.data + '/train/*.jpg'))}")
    print(f"Size of validation set: {len(glob(dm.data + '/val/*.jpg'))}")

    model = CNN_model().to(device)
    optimizer = Adam(model.parameters(),
                     lr=settings["learning_rate"],
                     weight_decay=settings["weight_decay"])

    loss_function = nn.CrossEntropyLoss()
    num_epochs = settings["num_epochs"]

