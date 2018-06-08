# ---------------------------------------------------------------
# Train YOLO Person Search
#
# Author: Liangqi Li
# Creating Date: May 17, 2018
# Latest rectified: Jun 8, 2018
# ---------------------------------------------------------------
import os
import argparse
import time

import yaml
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import dataset
from __init__ import clock_non_return
from darknet import Darknet19


def parse_args():
    "Parse input arguments"

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--epochs', default=450, type=int)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)
    parser.add_argument('--pre_model', default='', type=str)

    args = parser.parse_args()

    return args


def cuda_mode(args):
    """Set cuda"""
    if torch.cuda.is_available() and '-1' not in args.gpu_ids:
        cuda = True
        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)

        num_gpus = len(gpu_ids)

        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
    else:
        cuda = False
        num_gpus = 0

    return cuda, num_gpus


@clock_non_return
def main():

    opt = parse_args()
    use_cuda, num_gpus = cuda_mode(opt)

    train_list = '2012_train.txt'
    test_list = '2007_test.txt'
    num_samples = 5717
    num_workers = 10
    batch_size = opt.bs
    lr = opt.lr
    momentum = 0.9
    decay = 0.0005
    steps = [-1, 500, 40000, 60000]
    scales = [0.1, 10., 0.1, 0.1]

    # Training parameters
    max_epochs = opt.epochs
    seed = int(time.time())
    eps = 1e-5
    save_interval = 10  # epochs
    dot_interval = 70  # batches

    # Test parameters
    conf_thresh = .25
    nms_thresh = .4
    iou_thresh = .5

    save_dir = opt.out_dir
    print('Trained models will be save to', os.path.abspath(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.manual_seed(seed)

    pre_model = opt.pre_model
    model = Darknet19(pre_model)

    init_width = 416
    init_height = 416
    init_epoch = 0

    kwargs = {'num_workers': num_workers, 'pin_memory': True} \
        if use_cuda else {}
    test_loader = DataLoader(dataset.listDataset(
        test_list, shape=(init_width, init_height), shuffle=False,
        transform=transforms.Compose([transforms.ToTensor()]), train=False),
        batch_size=batch_size, shuffle=False, **kwargs)

    if use_cuda:
        if num_gpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >=0 :
            params += [{'params': [value], 'weight_decay': 0.}]
        else:
            params += [{'params': [value], 'weight_decay': decay * batch_size}]
    optimizer = optim.SGD(model.parameters(), lr=lr/batch_size,
                          momentum=momentum, dampening=0,
                          weight_decay=decay*batch_size)
