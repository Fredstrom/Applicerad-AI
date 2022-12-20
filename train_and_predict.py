from settings import settings
from datamanager import DataManager
from model import CNN_model

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.autograd import Variable
from glob import glob
from PIL import Image, ImageDraw, ImageFont


def model_train():
    dm = DataManager()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---" * 10)
    print(f"You are using: {device}")
    print("---" * 10)
    train_loader = dm.train_loader
    val_loader = dm.val_loader

    train_size = len(glob(dm.data + '/train/**/*.jpg'))
    val_size = len(glob(dm.data + '/val/**/*.jpg'))
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
        train_loss = train_loss / train_size * 100

        model.eval()
        val_accuracy = 0
        for idx, (img, label) in enumerate(val_loader):
            if torch.cuda.is_available():
                img = Variable(img.cuda())
                label = Variable(label.cuda())

            output = model(img)
            _, prediction = torch.max(output.data, 1)
            val_accuracy += int(torch.sum(label.data == prediction))
        val_accuracy = val_accuracy / val_size * 100

        print(f"Epoch: {epoch} / {num_epochs}")
        print(f"Train Loss: {train_loss:.0f}, Train Accuracy: {train_accuracy:.0f}%, Test Accuracy: {val_accuracy:.0f}%")
        if val_accuracy > best_accuracy:
            torch.save(model.state_dict(), f'Model_results/model_{val_accuracy:.0f}.pth')
            best_accuracy = val_accuracy
            print(f"New best: Saved model as 'model_{val_accuracy:.0f}.pth!")
        print(f"Best accuracy: {best_accuracy:.0f}%")
        print("------" * 10)


def model_predict(file_name="*"):
    module = CNN_model()
    module.load_state_dict(torch.load(settings["model_weights"]))
    module.eval()

    if file_name != "*":
        i = Image.open(settings['test_path'] + file_name)
        predictions = predict_image(module, settings['test_path'] + file_name)
        fnt = ImageFont.truetype("arial.ttf", 38)
        txt = ImageDraw.Draw(i)
        txt.text((120, 50), predictions, font=fnt, fill=(0, 0, 0))
        i.show()

    else:
        predictions = [predict_image(module, file) for file in glob(settings['test_path'] + file_name)]
        print(predictions)
        return predictions


def predict_image(model, image_path: str) -> str:
    # Load and preprocess the image
    image = Image.open(image_path)
    preprocess = transforms.Compose([transforms.Resize(settings["image_size"]), transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image_tensor = preprocess(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    label = prediction.argmax().item()
    label = "Not Algae" if label == 0 else "Algae"
    # Return the label(s) as a list of strings
    return label
