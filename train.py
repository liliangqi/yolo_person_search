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
from torch.autograd import Variable
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


def adjust_learning_rate(optimizer, batch, lr, batch_size, config):
    """Set the learning rate to the initial LR decayed by 10 every 30 epochs"""

    steps = config['lr_change_steps']
    scales = config['lr_change_scales']

    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size

    return lr


def train(model, epoch, train_loader, optimizer, opt, config, pro_bs,
          save_dir, save_interval):
    """Train the model"""

    lr = adjust_learning_rate(optimizer, pro_bs, opt.lr, opt.bs, config)
    print('epoch {}, processed {} samples, lr {}'.format(
        epoch, epoch * len(train_loader.dataset), lr))
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, pro_bs, opt.lr, opt.bs, config)
        pro_bs += 1

        data = data.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

        # Save the trained model
        if (epoch + 1) % save_interval == 0:
            save_name = os.path.join(
                save_dir, 'darknet19_{}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), save_name)

    return pro_bs


@clock_non_return
def main():

    opt = parse_args()
    use_cuda, num_gpus = cuda_mode(opt)
    with open('config.yml', 'r') as f:
        config = yaml.load(f)

    data_dir = opt.data_dir
    anno_dir = os.path.join(data_dir, 'annotations_cache')

    train_list = '2012_train.txt'
    test_list = '2007_test.txt'
    train_list_path = os.path.join(anno_dir, train_list)
    test_list_path = os.path.join(anno_dir, test_list)

    num_samples = 5717
    num_workers = config['train_num_workers']
    batch_size = opt.bs
    lr = opt.lr
    momentum = config['momentum']
    decay = config['decay']

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
        test_list_path, shape=(init_width, init_height), shuffle=False,
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

    pro_bs = 0
    for epoch in range(max_epochs):
        train_loader = DataLoader(dataset.listDataset(
            train_list_path, shape=(init_width, init_height), shuffle=True,
            transform=transforms.Compose([transforms.ToTensor()]), train=True,
            seen=0, batch_size=batch_size, num_workers=num_workers),
            batch_size=batch_size, shuffle=False, **kwargs)
        pro_bs = train(model, epoch, train_loader, optimizer, opt, config,
                       pro_bs, save_dir, save_interval)


if __name__ == '__main__':

    main()
