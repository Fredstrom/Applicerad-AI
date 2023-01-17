import os

from PIL.Image import Image

from settings import settings
from nn_model.datamanager import DataManager
from nn_model.model import CNN_model

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.autograd import Variable
import glob
from PIL import Image
from datetime import datetime


def model_train():
    dm = DataManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---" * 10)
    print(f"You are using: {device}")
    print("---" * 10)
    train_loader = dm.train_loader
    val_loader = dm.val_loader

    train_size = len(glob.glob(dm.data + '/train/**/*.jpg'))
    val_size = len(glob.glob(dm.data + '/val/**/*.jpg'))
    print(f"Size of training set: {train_size}")
    print(f"Size of validation set: {val_size}")
    print("---" * 10)

    model = CNN_model().to(device)
    optimizer = Adam(model.parameters(),
                     lr=settings["learning_rate"],
                     weight_decay=settings["weight_decay"])

    loss_function = nn.CrossEntropyLoss()
    num_epochs = settings["num_epochs"]

    best_accuracy = 0
    for epoch in range(settings["num_epochs"]):
        model.train()
        train_accuracy = 0
        train_loss = 0

        for idx, (img, label) in enumerate(train_loader):
            if torch.cuda.is_available():
                img = Variable(img.cuda())
                label = Variable(label.cuda())
            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data * img.size(0)
            _, prediction = torch.max(output.data, 1)

            train_accuracy += int(torch.sum(label.data == prediction))
        train_accuracy = train_accuracy / train_size * 100
        train_loss = (train_loss / train_size) * 100

        model.eval()
        val_accuracy = 0
        for idx, (img, label) in enumerate(val_loader):
            if torch.cuda.is_available():
                img = Variable(img.cuda())
                label = Variable(label.cuda())

            output = model(img)
            _, prediction = torch.max(output.data, 1)
            val_accuracy += int(torch.sum(label.data == prediction))
        val_accuracy = (val_accuracy / val_size) * 100

        print(f"Epoch: {epoch} / {num_epochs}")
        print(f"Train Loss: {train_loss:.0f}, Train Accuracy: {train_accuracy:.0f}%, Test Accuracy: {val_accuracy:.0f}%")
        if val_accuracy > best_accuracy:
            if not os.path.exists("Model_results/"):
                os.mkdir("Model_results/")
            torch.save(model.state_dict(), f'Model_results/model_{val_accuracy:.0f}.pth')
            best_accuracy = val_accuracy
            print(f"New best: Saved model as 'model_{val_accuracy:.0f}.pth!")
        print(f"Best accuracy: {best_accuracy:.0f}%")
        print("------" * 10)


def predict_image(folder, file, location="Not submitted"):
    if folder != settings["test_path"]:
        image_path = folder + file
    else:
        image_path = file
    model = CNN_model()
    model.load_state_dict(torch.load(settings["model_weights"]))
    model.eval()

    image = Image.open(image_path)
    preprocess = transforms.Compose([transforms.Resize(settings["image_size"]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image_tensor = preprocess(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
        confidence = torch.softmax(prediction, dim=1)

        confidence, label = torch.max(confidence, dim=1)
    label = prediction.argmax().float()
    save_results(label, confidence, file, location)
    return ("Algae", f'{int(confidence.item() * 100)}%') if label == 1 \
        else ("Not Algae", f'{int(confidence.item() * 100)}%')


def save_results(label, confidence, file, location):
    if not os.path.exists(settings["log_path"]):
        os.makedirs(settings["log_path"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'{settings["log_path"]}logs.csv', "a+") as logs:
        logs.write('\n')

        logs.write(f'{file}, '
                   f'{"Algae" if int(label) == 1 else "Not Algae"}, '
                   f'{confidence.item() * 100:.0f}%, '
                   f'{location}, '
                   f'{timestamp}')


def predict_folder(folder):
    print(folder)
    predictions = [predict_image(folder, image) for image in glob.glob(f"{folder}/*.jpg")]
    print(predictions)

# TODO: Add a test function that shows incorrect predictions (prediction, label, image).


