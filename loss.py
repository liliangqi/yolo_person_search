# ---------------------------------------------------------------
# Loss for YOLO Person Search
#
# Author: Liangqi Li
# Creating Date: Jun 8, 2018
# Latest rectified: Jun 9, 2018
# ---------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import convert2cpu, bbox_ious, bbox_iou


def build_targets(pred_boxes, target, anchors, num_anchors, n_h, n_w, config):
    n_b = target.size(0)
    n_a = num_anchors
    anchor_step = 2
    noobject_scale = config['noobject_scale']
    object_scale = config['object_scale']
    sil_thresh = config['sil_thresh']

    conf_mask = torch.ones(n_b, n_a, n_h, n_w) * noobject_scale
    coord_mask = torch.zeros(n_b, n_a, n_h, n_w)
    cls_mask = torch.zeros(n_b, n_a, n_h, n_w)
    tx = torch.zeros(n_b, n_a, n_h, n_w)
    ty = torch.zeros(n_b, n_a, n_h, n_w)
    tw = torch.zeros(n_b, n_a, n_h, n_w)
    th = torch.zeros(n_b, n_a, n_h, n_w)
    tconf = torch.zeros(n_b, n_a, n_h, n_w)
    tcls = torch.zeros(n_b, n_a, n_h, n_w)

    n_all_anchors = n_a * n_h * n_w
    n_pixels = n_h * n_w

    for b in range(n_b):
        cur_pred_boxes = pred_boxes[b*n_all_anchors: (b+1)*n_all_anchors].t()
        cur_ious = torch.zeros(n_all_anchors)
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1] * n_w
            gy = target[b][t*5+2] * n_h
            gw = target[b][t*5+3] * n_w
            gh = target[b][t*5+4] * n_h
            cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(
                n_all_anchors, 1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(
                cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask[b][cur_ious > sil_thresh] = 0

    tx.fill_(0.5)
    ty.fill_(0.5)
    tw.zero_()
    th.zero_()
    coord_mask.fill_(1)

    n_gt = 0
    n_correct = 0
    for b in range(n_b):
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            n_gt += 1
            best_iou = 0.0
            best_n = -1
            gx = target[b][t*5+1] * n_w
            gy = target[b][t*5+2] * n_h
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3] * n_w
            gh = target[b][t*5+4] * n_h
            gt_box = [0, 0, gw, gh]
            for n in range(n_a):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*n_all_anchors+best_n*n_pixels+gj*n_w+gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * n_w - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * n_h - gj
            tw[b][best_n][gj][gi] = math.log(
                gw / anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(
                gh / anchors[anchor_step*best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            if iou > 0.5:
                n_correct = n_correct + 1

    return n_gt, n_correct, coord_mask, conf_mask, cls_mask, tx, ty, tw, th,\
           tconf, tcls


def region_loss(output, target, config):

    anchors = config['anchors']
    n_b = output.data.size(0)
    n_a = len(anchors) // 2
    n_c = config['num_classes']
    n_h = output.data.size(2)
    n_w = output.data.size(3)

    output = output.view(n_b, n_a, (5 + n_c), n_h, n_w)
    x = F.sigmoid(output.index_select(2, Variable(
        torch.cuda.LongTensor([0]))).view(n_b, n_a, n_h, n_w))
    y = F.sigmoid(output.index_select(2, Variable(
        torch.cuda.LongTensor([1]))).view(n_b, n_a, n_h, n_w))
    w = output.index_select(2, Variable(
        torch.cuda.LongTensor([2]))).view(n_b, n_a, n_h, n_w)
    h = output.index_select(2, Variable(
        torch.cuda.LongTensor([3]))).view(n_b, n_a, n_h, n_w)
    conf = F.sigmoid(output.index_select(2, Variable(
        torch.cuda.LongTensor([4]))).view(n_b, n_a, n_h, n_w))
    cls = output.index_select(2, Variable(
        torch.linspace(5, 5+n_c-1, n_c).long().cuda()))
    cls = cls.view(n_b*n_a, n_c, n_h*n_w).transpose(1, 2).contiguous().view(
        n_b*n_a*n_h*n_w, n_c)

    pred_boxes = torch.cuda.FloatTensor(4, n_b*n_a*n_h*n_w)
    grid_x = torch.linspace(0, n_w-1, n_w).repeat(n_h, 1).repeat(
        n_b*n_a, 1, 1).view(n_b*n_a*n_h*n_w).cuda()
    grid_y = torch.linspace(0, n_h-1, n_h).repeat(n_w, 1).t().repeat(
        n_b*n_a, 1, 1).view(n_b*n_a*n_h*n_w).cuda()
    anchor_w = torch.Tensor(anchors).view(n_a, 2).index_select(
        1, torch.LongTensor([0])).cuda()
    anchor_h = torch.Tensor(anchors).view(n_a, 2).index_select(
        1, torch.LongTensor([1])).cuda()
    anchor_w = anchor_w.repeat(n_b, 1).repeat(1, 1, n_h*n_w).view(
        n_b*n_a*n_h*n_w)
    anchor_h = anchor_h.repeat(n_b, 1).repeat(1, 1, n_h*n_w).view(
        n_b*n_a*n_h*n_w)

    pred_boxes[0] = x.data + grid_x
    pred_boxes[1] = y.data + grid_y
    pred_boxes[2] = torch.exp(w.data) * anchor_w
    pred_boxes[3] = torch.exp(h.data) * anchor_h
    pred_boxes = convert2cpu(
        pred_boxes.transpose(0, 1).contiguous().view(-1 ,4))

    n_gt, n_correct, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,\
    tcls = build_targets(pred_boxes, target.data, anchors, n_a, n_h, n_w,
                         config)
    cls_mask = (cls_mask == 1)
    n_proposals = int((conf > .25).sum().data[0])

    tx = Variable(tx.cuda())
    ty = Variable(ty.cuda())
    tw = Variable(tw.cuda())
    th = Variable(th.cuda())
    tconf = Variable(tconf.cuda())
    tcls = Variable(tcls.view(-1)[cls_mask].long().cuda())

    coord_mask = Variable(coord_mask.cuda())
    conf_mask = Variable(conf_mask.cuda().sqrt())
    cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, n_c).cuda())
    cls = cls[cls_mask].view(-1, n_c)

    coord_scale = config['coord_scale']
    class_scale = config['class_scale']
    loss_x = coord_scale * nn.MSELoss(size_average=False)(
        x * coord_mask, tx * coord_mask) / 2
    loss_y = coord_scale * nn.MSELoss(size_average=False)(
        y * coord_mask, ty * coord_mask) / 2
    loss_w = coord_scale * nn.MSELoss(size_average=False)(
        w * coord_mask, tw * coord_mask) / 2
    loss_h = coord_scale * nn.MSELoss(size_average=False)(
        h * coord_mask, th * coord_mask) / 2
    loss_conf = nn.MSELoss(size_average=False)(
        conf * conf_mask, tconf * conf_mask) / 2
    loss_cls = class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
    print('nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f,'
          ' conf %f, cls %f, total %f' % (
        n_gt, n_correct, n_proposals, loss_x.data[0], loss_y.data[0],
        loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0],
        loss.data[0]))

    return loss