import numpy as np
import torch.utils.data as data

class faceData(data.Dataset):
    def __init__(self, data_path='Data/fer2013.csv', mode='train', transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.load()

    def load(self):
        if self.mode == 'train':
            usage = 'Training'
        else:
            usage = 'PrivateTest'
        self.data = np.load('Data/%s_data.npy' % usage)
        # self.data = self.data[:,np.newaxis, :, :]
        self.label = np.load('Data/%s_label.npy' % usage)
        print(self.data.shape, self.label.shape)
    
    def __getitem__(self, item):
        img, label = self.data[item], self.label[item]
        img = img[np.newaxis, :, :]
        # img = np.transpose((1, 2, 0))
        if self.transform != None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return self.label.shape[0]
        