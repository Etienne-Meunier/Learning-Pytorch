import torch
import os
import pandas as pd
from skimage import io,transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms, utils
from torchvision import transforms
import torch


class CsvClassificationDataset(Dataset) :
    """
    Dataset with a csv indication label for classification
        Path Image          Label
        .../1.png             0
        .../2.png             1
        .../3.png             0
    """

    def __init__(self,csv_file,root_dir=None,transform=None) :
        """
        Args :
            csv_file (string) : Path of the csv file with labels
            root_dir (string, optional) : Base dir to add to the path of images
            transform (callable, optional) : Optional transform to be applied
                on the sample
        """

        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) :
        """
            Return the len of the dataset overwrite function from Dataset
        """
        return len(self.labels)

    def __getitem__(self, idx) :
        """
            Return an item of the dataset depending of the index

        Args :
            idx : index of the item to return

        Returns :
            Sample dict as {'image','label'}
        """
        if torch.is_tensor(idx) :
            # We need a list for pandas operations
            idx = idx.tolist()
        # Load the image
        img_name = self.labels.iloc[idx,0]
        if self.root_dir:
            img_name = os.path.join(self.root_dir,img_name)
        image = default_loader(img_name)
        if self.transform :
            image = self.transform(image)
        label = self.labels.iloc[idx,1]
        sample = {'image':image, 'label':label}
        return sample


if __name__ == '__main__':
    # Quick test
    csv_train ='/home/etienne/Dropbox/Labels/RetinaNet_Labels_CRA/retina_classif_train.csv'
    tr = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
    csv_dataset = CsvClassificationDataset(csv_file=csv_train,transform=tr)
    dataloader = torch.utils.data.DataLoader(csv_dataset,batch_size=4,shuffle=True)
