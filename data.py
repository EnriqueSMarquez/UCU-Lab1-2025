import torch
import os
from PIL import Image
import pandas as pd

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, df, transforms=None):
        self.images_path = images_path
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index):
        # LOGICA DE CARGADO DE IMAGEN PARA EL INDICE INDEX
        # LOGICA DE CARGADO DE LABEL PARA EN INDECE INDEX
        image_name = str(self.df.loc[index, 'image_name']) #0.jpg
        label = int(self.df.loc[index, 'label'])  # 5
        image_path = os.path.join(self.images_path, image_name)
        image = Image.open(image_path)

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.df)