import os
from PIL import Image
import time


class DataManager:
    def __init__(self, infile, size):
        self.in_file = infile
        self.out_file = infile + "_jpg"
        self.size = size


    def convert_to_jpg(self) -> None:
        """
        Converts and resizes the images from your in_file.

        dynamically creates an out_folder matching the structure of the file the data was originally stored in.
        also resizes, and saves the images as the desired ratio. (Default 200, 200)
        """
        train_split = 0.8
        start_time = time.monotonic()
        accepted_formats = [".jpg", ".png", ".jpeg", "bmp"]

        if not os.path.exists(self.out_file):
            os.makedirs(self.out_file)

        for category in ["train", "val"]:
            if not os.path.exists(os.path.join(self.out_file, category)):
                os.makedirs(os.path.join(self.out_file, category))

        for root, _, images in os.walk(self.in_file):  # os.walk goes through a path from the root, step by step.

            # splits the input_data into train_split:rest and saves the train-set in the train folders
            for i in images[:int(len(images) * train_split)]:
                label = root.split("\\")[1]  # find the folder containing our images
                out_path = os.path.join(self.out_file, "train", label)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                file_name, file_ext = os.path.splitext(i)
                if file_ext.lower() in accepted_formats:
                    img = Image.open(f"{self.in_file}/{label}/{i}")
                    img = img.resize(self.size)
                    img.save(f'{out_path}/{file_name}.jpg')

            # Takes the remaining images and saves them under the val folders
            for i in images[int(len(images) * train_split):]:
                label = root.split("\\")[1]  # find the folder containing our images
                out_path = os.path.join(self.out_file, "val", label)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                file_name, file_ext = os.path.splitext(i)
                if file_ext.lower() in accepted_formats:
                    img = Image.open(f"{self.in_file}/{label}/{i}")
                    img = img.resize(self.size)
                    img.save(f'{out_path}/{file_name}.jpg')

        end_time = time.monotonic()
        runtime = end_time - start_time

        print(f'The resized pictures can be found at: {self.out_file}')
        print(f'The operation took {runtime:.2f} seconds.')
