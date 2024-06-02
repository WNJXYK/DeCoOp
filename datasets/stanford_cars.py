import os
from scipy.io import loadmat
from torch.utils.data import Dataset
from utils.util_data import (read_split, 
                             subsample_classes, 
                             generate_fewshot_dataset, 
                             read_image,
                             Datum,
                             read_json, 
                             write_json,
                             get_lab2cname)

template = ['a photo of a {}.']

class StanfordCars(Dataset):
    dataset_dir = 'stanford_cars'

    def __init__(self, root, num_shots, subsample, transform=None, type='train', seed=0):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # problem to fix
        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        train = generate_fewshot_dataset(train, num_shots=num_shots, seed=seed)

        self.subsample = subsample
        train, val, test = subsample_classes(train, val, test, subsample=self.subsample)
        dataset = {'train' : train, 'val' : val, 'test' : test}
        self.data_source = dataset[type]
        self.label2cname, self.cname2label, self.classnames = get_lab2cname(self.data_source)
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        image = read_image(item.impath)
        if self.transform:
            image = self.transform(image)
        
        return image, item.label, item.classname
    
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)['annotations'][0]
        meta_file = loadmat(meta_file)['class_names'][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]['fname'][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]['class'][0, 0]
            label = int(label) - 1 # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            classname = ' '.join(names)
            item = Datum(
                impath=impath,
                label=label,
                classname=classname
            )
            items.append(item)
        
        return items
    