# ---------------------------------------------------------------
# Train YOLO Person Search
#
# Author: Liangqi Li
# Creating Date: May 17, 2018
# Latest rectified: Jun 10, 2018
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
from loss import region_loss
from utils import get_region_boxes, nms, bbox_iou


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--epochs', default=450, type=int)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
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
        output = model(data)
        loss = region_loss(output, target, config)
        loss.backward()
        optimizer.step()

        # Save the trained model
        if (epoch + 1) % save_interval == 0:
            save_name = os.path.join(
                save_dir, 'darknet19_{}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), save_name)

    return pro_bs


def test(model, test_loader, config):
    """Test the model during training"""

    def truths_length(truth):
        for k in range(50):
            if truth[k][1] == 0:
                return k

    model.eval()
    num_classes = config['num_classes']
    anchors = config['anchors']
    num_anchors = len(anchors) // 2
    conf_thresh = config['conf_thresh']
    nms_thresh = config['nms_thresh']
    iou_thresh = config['iou_thresh']
    eps = 1e-5
    total = 0.
    proposals = 0.
    correct = 0.

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors,
                                     num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

            total += num_gts
            for l in range(len(boxes)):
                if boxes[l][4] > conf_thresh:
                    proposals += 1
            for l in range(num_gts):
                box_gt = [truths[l][1], truths[l][2], truths[l][3],
                          truths[l][4], 1., 1., truths[l][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct += 1

    precision = 1. * correct / (proposals + eps)
    recall = 1. * correct / (total + eps)
    fscore = 2. * precision * recall / (precision + recall + eps)
    print('precision: {}, recall: {}, fscore: {}'.format(
        precision, recall, fscore))


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

    num_workers = config['train_num_workers']
    batch_size = opt.bs
    lr = opt.lr
    momentum = config['momentum']
    decay = config['decay']

    # Training parameters
    max_epochs = opt.epochs
    seed = int(time.time())
    save_interval = 10  # epochs

    save_dir = opt.out_dir
    print('Trained models will be save to', os.path.abspath(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.manual_seed(seed)

    pre_model = opt.pre_model
    model = Darknet19(pre_model)

    init_width = 416
    init_height = 416

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
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
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
        test(model, test_loader, config)


if __name__ == '__main__':

    main()
