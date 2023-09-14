import torchvision.transforms as transforms
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_digits_domain_skew_loaders
import torch.utils.data as data
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNet import resnet10, resnet12
from backbone.efficientnet import EfficientNetB0
from backbone.mobilnet_v2 import MobileNetV2
from torchvision.datasets import MNIST, SVHN, ImageFolder, DatasetFolder, USPS
# from backbone.BN_models import AlexNetBN, ConvNet, 
from backbone.utils import get_network
"""
As you can see all the models have been pulled from the module named backbone
"""
#------------

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
# from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
import fedbn_digits as data_utils

#---------
"""
batch_size and percent are the only two variables.
"""
def prepare_data(percent, batch_size):
    # Prepare data
    # the number ofop channels is 3!
    image_size = [32, 32] # important resizing operation! necessary for each to work!
    transform_mnist = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = data_utils.DigitsDataset(data_path="./data/MNIST", channels=1, percent=percent, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path="./data/MNIST", channels=1, percent=percent, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = data_utils.DigitsDataset(data_path='./data/SVHN', channels=3, percent=percent,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path='./data/SVHN', channels=3, percent=percent,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = data_utils.DigitsDataset(data_path='./data/USPS', channels=1, percent=percent,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path='./data/USPS', channels=1, percent=percent,  train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset     = data_utils.DigitsDataset(data_path='./data/SynthDigits/', channels=3, percent=percent,  train=True,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path='./data/SynthDigits/', channels=3, percent=percent,  train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset     = data_utils.DigitsDataset(data_path='./data/MNIST_M/', channels=3, percent=percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = data_utils.DigitsDataset(data_path='./data/MNIST_M/', channels=3, percent=percent,  train=False, transform=transform_mnistm)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=batch_size,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=batch_size, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=batch_size,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=batch_size, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=batch_size,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=batch_size, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=batch_size,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=batch_size, shuffle=False)

    # change the size of the loaders here, only 4 are being used.
    train_loaders = [mnist_train_loader, usps_train_loader,  svhn_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader,  usps_test_loader, svhn_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders

class MyDigits(data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()
        
    def __build_truncated_dataset__(self):
        if self.data_name == 'mnist':
            dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'usps':
            dataobj = USPS(self.root, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'svhn':
            if self.train:
                dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
        return dataobj

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.dataset[index]
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/train/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/val/', self.transform, self.target_transform)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaDigits(FederatedDataset):
    NAME = 'fl_digits'
    SETTING = 'domain_skew' # it is a domain_skew setup
    DOMAINS_LIST = ['mnist', 'usps', 'svhn', 'syn']
    percent_dict = {'mnist': 0.0023, 'usps': 0.013, 'svhn': 0.13, 'syn': 0.23}
    # 0.0023,0.013,0.13,0.305
    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    PERCENT=0.1
    # BATCH_SIZE  = 32 # I DO NOT KNOW
    Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    Singel_Channel_Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])

    def get_data_loaders(self, selected_domain_list=[]):

        # using_list = self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list

        # nor_transform = self.Nor_TRANSFORM
        # sin_chan_nor_transform = self.Singel_Channel_Nor_TRANSFORM
        # train_dataset_list = []
        # test_dataset_list = []
        # test_transform = transforms.Compose(
        #     [transforms.Resize((32, 32)), transforms.ToTensor(), self.get_normalization_transform()])
        # sin_chan_test_transform = transforms.Compose(
        #     [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), self.get_normalization_transform()])
        # for _, domain in enumerate(using_list):
        #     if domain == 'syn':
        #         train_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=True,
        #                                            transform=nor_transform)
        #     else:
        #         if domain in ['mnist', 'usps']:
        #             train_dataset = MyDigits(data_path(), train=True,
        #                                      download=True, transform=sin_chan_nor_transform, data_name=domain)
        #         else:
        #             train_dataset = MyDigits(data_path(), train=True,
        #                                      download=True, transform=nor_transform, data_name=domain)
        #     train_dataset_list.append(train_dataset)

        # for _, domain in enumerate(self.DOMAINS_LIST):
        #     if domain == 'syn':
        #         test_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=False,
        #                                           transform=test_transform)
        #     else:
        #         if domain in ['mnist', 'usps']:
        #             test_dataset = MyDigits(data_path(), train=False,
        #                                     download=True, transform=sin_chan_test_transform, data_name=domain)
        #         else:

        #             test_dataset = MyDigits(data_path(), train=False,
        #                                     download=True, transform=test_transform, data_name=domain)

        #     test_dataset_list.append(test_dataset)
        # traindls, testdls = partition_digits_domain_skew_loaders(train_dataset_list, test_dataset_list, self)
        #----------------------------------------------------------------
        # over here I will rewrite the important terms of the equation, i.e. traindls and testdls. They will come from the technique that was mentioned in CYH's work
        # here encode the fedbn terms!
        # construct an args variable here that has those variables.
        self.BATCH_SIZE = self.args.local_batch_size
        traindls, testdls  = prepare_data(self.PERCENT, self.BATCH_SIZE) 
        # it = iter(testdls[0])
        # x, y = next(it)
        # print(x.shape)
        return traindls, testdls

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaDigits.Nor_TRANSFORM])
        return transform

    # this is the backbone that is used during the training phase
    @staticmethod
    def get_backbone(parti_num, names_list):
        # may have to change this to change the backbone
        nets_dict = {'resnet10': resnet10, 'resnet12': resnet12, 'efficient': EfficientNetB0, 'mobilnet': MobileNetV2, 
                     'AlexNetBN': get_network, 'ConvNetBN': get_network, 'VGG11BN':get_network,
                     'AlexNet':get_network, 'ConvNet': get_network}
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                nets_list.append(resnet12(FedLeaDigits.N_CLASS)) # simply get a resent 12 with the number of classes mentioned as an argument
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                if(nets_dict[net_name] == get_network):
                    print(f"Model {j} is {net_name}")
                    print(net_name, FedLeaDigits.N_CLASS)
                    nets_list.append(nets_dict[net_name](net_name, 3, FedLeaDigits.N_CLASS, (32, 32)))
                else:
                    nets_list.append(nets_dict[net_name](FedLeaDigits.N_CLASS)) # style of calling is going to be different
        return nets_list 

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform
