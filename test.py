# ---------------------------------------------------------------
# Test YOLO Person Search
#
# Author: Liangqi Li
# Creating Date: Jun 10, 2018
# Latest rectified: Jun 10, 2018
# ---------------------------------------------------------------
import os
import argparse
import time

import yaml
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import xml.etree.ElementTree as ET

import dataset
from __init__ import clock_non_return
from darknet import Darknet19
from utils import get_region_boxes, nms, get_image_size


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--bs', default=2, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--trained_model', default='', type=str)
    parser.add_argument('--out_dir', default='./output', type=str)

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


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def test(model, test_loader, test_files, config):
    """Test the result"""

    # TODO: use config.yml
    conf_thresh = 0.005
    nms_thresh = 0.45
    num_classes = config['num_classes']
    anchors = config['anchors']
    num_anchors = len(anchors) // 2

    line_id = -1
    results = {}
    for i in range(num_classes):
        results[i] = []

    start = time.time()
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        batch_boxes = get_region_boxes(
            output, conf_thresh, num_classes, anchors, num_anchors, 0, 1)
        for i in range(output.size(0)):
            line_id += 1
            im_id = os.path.basename(test_files[line_id]).split('.')[0]
            width, height = get_image_size(test_files[line_id])
            boxes = batch_boxes[i]
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2] / 2) * width
                y1 = (box[1] - box[3] / 2) * height
                x2 = (box[0] + box[2] / 2) * width
                y2 = (box[1] + box[3] / 2) * height

                det_conf = box[4]
                for j in range((len(box) - 5) // 2):
                    cls_conf = box[5 + 2*j]
                    cls_id = box[6 + 2*j]
                    prob = det_conf * cls_conf
                    results[cls_id].append((im_id, prob, x1, y1, x2, y2))
        end = time.time()
        print('Batch {}/{}, time cost: {:.4f}s per image'.format(
            batch_idx + 1, len(test_loader),
            (end - start) / ((batch_idx + 1) * output.size(0))))

    return results


def evaluate(results, data_dir, class_names):
    """Evaluate the detection results"""

    anno_path = os.path.join(data_dir, 'VOC2007', 'Annotations', '{:s}.xml')
    imset_file = os.path.join(data_dir, 'VOC2007', 'ImageSets/Main/test.txt')

    with open(imset_file, 'r') as f:
        lines = f.readlines()
    im_names = [x.strip() for x in lines]

    recs = {}
    for i, im_name in enumerate(im_names):
        recs[im_name] = parse_rec(anno_path.format(im_name))
        if i % 100 == 0:
            print('Reading annotation for {}/{}'.format(
                i + 1, len(im_names)))

    aps = []
    ov_thresh = 0.5

    for cls_id, cls in enumerate(class_names):

        class_recs = {}
        n_pos = 0
        for im_name in im_names:
            objs = [obj for obj in recs[im_name] if obj['name'] == cls]
            bbox = np.array([x['bbox'] for x in objs])
            difficult = np.array(
                [x['difficult'] for x in objs]).astype(np.bool)
            det = [False] * len(objs)
            n_pos += sum(~difficult)
            class_recs[im_name] = {
                'bbox': bbox, 'difficult': difficult, 'det': det}

        # Read dets
        split_lines = results[cls_id]
        im_ids = [x[0] for x in split_lines]
        confidence = np.array([float(x[1]) for x in split_lines])
        det_bbox = np.array([[float(z) for z in x[2:]] for x in split_lines])

        # Sort by confidence
        sorted_inds = np.argsort(-confidence)
        det_bbox = det_bbox[sorted_inds, :]
        im_ids = [im_ids[x] for x in sorted_inds]

        # Go down dets and mark TPs and FPs
        nd = len(im_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            c_rec = class_recs[im_ids[d]]
            bb = det_bbox[d, :].astype(float)
            ov_max = -np.inf
            bb_gt = c_rec['bbox'].astype(float)

            if bb_gt.size > 0:
                ixmin = np.maximum(bb_gt[:, 0], bb[0])
                iymin = np.maximum(bb_gt[:, 1], bb[1])
                ixmax = np.minimum(bb_gt[:, 2], bb[2])
                iymax = np.minimum(bb_gt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # Union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bb_gt[:, 2] - bb_gt[:, 0] + 1.) *
                       (bb_gt[:, 3] - bb_gt[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ov_max = np.max(overlaps)
                j_max = np.argmax(overlaps)

            if ov_max > ov_thresh:
                if not c_rec['difficult'][j_max]:
                    if not c_rec['det'][j_max]:
                        tp[d] = 1.
                        c_rec['det'][j_max] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # Compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(n_pos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, True)

        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))

    print('Mean AP = {:.4f}'.format(np.mean(aps)))


@clock_non_return
def main():

    opt = parse_args()
    use_cuda, num_gpus = cuda_mode(opt)
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    class_names = config['class_names']

    data_dir = opt.data_dir
    anno_dir = os.path.join(data_dir, 'annotations_cache')
    test_file_path = os.path.join(anno_dir, '2007_test.txt')
    with open(test_file_path, 'r') as f:
        tmp_files = f.readlines()
        test_files = [item.rstrip() for item in tmp_files]

    model = Darknet19()
    model.load_trained_model(torch.load(opt.trained_model))
    if use_cuda:
        model.cuda()
    model.eval()

    init_width = 416
    init_height = 416
    test_dataset = dataset.listDataset(
        test_file_path, shape=(init_width, init_height), shuffle=False,
        transform=transforms.Compose([transforms.ToTensor()]))
    test_bs = opt.bs
    assert test_bs > 1
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False,
                             **kwargs)

    results = test(model, test_loader, test_files, config)
    evaluate(results, data_dir, class_names)


if __name__ == '__main__':

    main()
