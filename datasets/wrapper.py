import os
from torch.utils.data import Dataset
from utils.util_data import (read_split, 
                             subsample_classes, 
                             generate_fewshot_dataset, 
                             read_image,
                             read_json, 
                             write_json,
                             get_lab2cname)

template = ['a photo of {}, a type of food.']

class WrapperDataset(Dataset):
    def __init__(self, dataset, clip, score):
        self.data_source = dataset.data_source
        self.label2cname, self.cname2label, self.classnames = dataset.label2cname, dataset.cname2label, dataset.classnames
        self.transform = dataset.transform
        self.clip = clip
        self.score = score

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        image = read_image(item.impath)
        if self.transform:
            image = self.transform(image)
        clip = self.clip[idx]
        score = self.score[idx]
        
        return image, item.label, item.classname, clip, score