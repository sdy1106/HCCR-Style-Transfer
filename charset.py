# 2939x200x64x64 size of image dataset

import os

import torch
import torch.utils.data as data

import random
import numpy as np


class Charset(data.Dataset):
    def __init__(self, data_dir, file_name, ratio=0.8, is_train=True, single_size=16):
        self.is_train = is_train
        self.style_index = 0
        self.data = None
        self.single_size = single_size
        try:
            data = np.load(os.path.join(data_dir, file_name))
        except:
            print ('reading data error!')
            os._exit(0)
        self.character_num = data.shape[0]
        self.style_num = data.shape[1]
        self.train_character_num = int(self.character_num * ratio)
        self.train_style_num = int(self.style_num * ratio)
        self.test_character_num = self.character_num - self.train_character_num
        self.test_style_num = self.style_num - self.test_character_num

        if is_train:
            self.data = data[:self.train_character_num, :self.train_style_num, :, :]
        else:
            self.data = data[self.train_character_num:, self.train_style_num:, :, :]

    def __getitem__(self, index):
        content_index = index
        style_index = (index + random.randrange(self.train_style_num)) % self.train_style_num
        style_sample_index = range(self.train_character_num)
        content_sample_index = range(self.train_style_num)
        random.shuffle(style_sample_index)
        random.shuffle(content_sample_index)
        style_sample_index = style_sample_index[:self.single_size]
        content_sample_index = content_sample_index[:self.single_size]

        style_batch = self.data[style_sample_index, style_index, :, :]
        content_batch = self.data[content_index, content_sample_index, :, :]
        output = torch.stack([torch.from_numpy(style_batch), torch.from_numpy(content_batch)]).type(torch.FloatTensor)
        gt = torch.from_numpy(self.data[content_index, style_index]).type(torch.FloatTensor)
        #print (output.size())
        return output, gt

    def __len__(self):
        return self.data.shape[0]
