from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io

class KineticsImg(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, spatial_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.spatial_transform = spatial_transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        target = self.landmarks_frame.iloc[idx, 1:].as_matrix()

        if self.spatial_transform:
            sample = self.spatial_transform(image)

        return sample, target, target