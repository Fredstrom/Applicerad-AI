from torchvision.transforms import transforms
from settings import settings
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class DataManager:

    def __init__(self):
        self.transforms = transforms.Compose([transforms.Resize(settings["image_size"]),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                              ])
        self.data = settings["data_path"]
        self.train_loader = DataLoader(ImageFolder(f"{self.data}/train",
                                                   transform=self.transforms),
                                       batch_size=settings["batch_size"], shuffle=True)

        self.val_loader = DataLoader(ImageFolder(f"{self.data}/val",
                                                 transform=self.transforms),
                                     batch_size=settings["batch_size"],
                                     shuffle=True)
