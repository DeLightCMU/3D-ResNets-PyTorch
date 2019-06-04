from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

class KineticsImg(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, spatial_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.spatial_transform = spatial_transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.landmarks_frame.iloc[idx, 0])

        with open(img_name, 'rb') as f:
            with Image.open(f) as img:
                sample = img.convert('RGB')
        target = self.landmarks_frame.iloc[idx, 1:].values
        target = target.astype('int')

        if self.spatial_transform:
            sample = self.spatial_transform(sample)

        return sample, target, target