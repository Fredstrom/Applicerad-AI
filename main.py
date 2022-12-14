from torchvision.datasets import ImageFolder
import torch

def main():
    train_data = ImageFolder("data_jpg/train")
    val_data = ImageFolder("data_jpg/val")

    # CNN Training goes here #

if __name__ == "__main__":
    main()
