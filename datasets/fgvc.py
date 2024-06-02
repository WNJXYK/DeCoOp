import os
from torch.utils.data import Dataset
from utils.util_data import (read_split, 
                             subsample_classes, 
                             generate_fewshot_dataset, 
                             read_image,
                             Datum,
                             read_json, 
                             write_json,
                             get_lab2cname)

template = ['a photo of a {}, a type of aircraft.']

class FGVCAircraft(Dataset):

    dataset_dir = 'fgvc_aircraft'

    def __init__(self, root, num_shots, subsample, transform=None, type='train', seed=0):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        self.template = template

        classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, 'images_variant_train.txt')
        val = self.read_data(cname2lab, 'images_variant_val.txt')
        test = self.read_data(cname2lab, 'images_variant_test.txt')
        
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
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        
        return items