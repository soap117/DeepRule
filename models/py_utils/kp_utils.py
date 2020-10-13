import torch
import torch.nn as nn
import numpy as np
from .utils import convolution, residual


class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2


def make_merge_layer(dim):
    return MergeUp()


def make_tl_layer(dim):
    return None


def make_br_layer(dim):
    return None


def make_center_layer(dim):
    return None


def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)


def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


def make_inter_layer(dim):
    return residual(3, dim, dim)


def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _tranpose_and_gather_features(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _getGT(scores, GT, K):
    batch, cat, height, width = scores.size()
    topk_inds = GT[:, :, 2] * height * width + GT[:, :, 1] * width + GT[:, :, 0]
    topk_inds.view(1, -1)
    if topk_inds.shape[1] < K:
        tmp = torch.zeros((topk_inds.shape[0], K - topk_inds.shape[1]), dtype=topk_inds.dtype)
        topk_inds = torch.cat((topk_inds, tmp), 1)
    else:
        topk_inds = topk_inds[:, 0:K]
    topk_scores_, topk_inds_ = torch.topk(scores.view(batch, -1), K)
    topk_inds = topk_inds.cuda()
    topk_scores = torch.ones_like(topk_inds).float().cuda()
    topk_inds = topk_inds.long()
    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_xs_ = (topk_inds_ % width).int().float()
    print(topk_xs)
    print(topk_xs_)
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _decode(
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_tag_ = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag_ = tl_tag_.view(batch, K)
    br_tag_ = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag_ = br_tag_.view(batch, K)
    tl_regr_ = _tranpose_and_gather_feat(tl_regr, tl_inds)
    br_regr_ = _tranpose_and_gather_feat(br_regr, br_inds)

    tl_scores_ = tl_scores.view(1, batch, K)
    tl_tag_ = tl_tag_.view(1, batch, K)
    tl_clses_ = tl_clses.view(1, batch, K)
    tl_xs_ = tl_xs.view(1, batch, K)
    # print('_________________')
    # print(tl_xs_[0, 0])
    tl_ys_ = tl_ys.view(1, batch, K)
    tl_regr_ = tl_regr_.view(1, batch, K, 2)
    tl_xs_ += tl_regr_[:, :, :, 0]
    # print(tl_xs_[0, 0])
    tl_ys_ += tl_regr_[:, :, :, 1]
    br_scores_ = br_scores.view(1, batch, K)
    br_tag_ = br_tag_.view(1, batch, K)
    br_clses_ = br_clses.view(1, batch, K)
    br_xs_ = br_xs.view(1, batch, K)
    br_ys_ = br_ys.view(1, batch, K)
    br_regr_ = br_regr_.view(1, batch, K, 2)
    br_xs_ += br_regr_[:, :, :, 0]
    br_ys_ += br_regr_[:, :, :, 1]
    detections_tl = torch.cat([tl_scores_, tl_tag_, tl_clses_.float(), tl_xs_, tl_ys_], dim=0)
    detections_br = torch.cat([br_scores_, br_tag_, br_clses_.float(), br_xs_, br_ys_], dim=0)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    # print(tl_xs[0, :, 0])
    '''
    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
    '''
    # print(tl_xs[0, :, 0])
    # print('_________________')
    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections, detections_tl, detections_br


def _decode_pure(
        tl_heat, br_heat, tl_regr, br_regr,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    # print(tl_scores)
    tl_regr_ = _tranpose_and_gather_feat(tl_regr, tl_inds)
    br_regr_ = _tranpose_and_gather_feat(br_regr, br_inds)


    tl_scores_ = tl_scores.view(1, batch, K)
    tl_clses_ = tl_clses.view(1, batch, K)
    tl_xs_ = tl_xs.view(1, batch, K)
    # print('_________________')
    # print(tl_xs_[0, 0])
    tl_ys_ = tl_ys.view(1, batch, K)
    tl_regr_ = tl_regr_.view(1, batch, K, 2)
    tl_xs_ += tl_regr_[:, :, :, 0]
    # print(tl_xs_[0, 0])
    tl_ys_ += tl_regr_[:, :, :, 1]
    br_scores_ = br_scores.view(1, batch, K)
    br_clses_ = br_clses.view(1, batch, K)
    br_xs_ = br_xs.view(1, batch, K)
    br_ys_ = br_ys.view(1, batch, K)
    br_regr_ = br_regr_.view(1, batch, K, 2)
    br_xs_ += br_regr_[:, :, :, 0]
    br_ys_ += br_regr_[:, :, :, 1]
    detections_tl = torch.cat([tl_scores_, tl_clses_.float(), tl_xs_, tl_ys_], dim=0)
    detections_br = torch.cat([br_scores_, br_clses_.float(), br_xs_, br_ys_], dim=0)

    return detections_tl, detections_br

def _decode_line_cls(ps_results, ng_results):
    if torch.numel(ps_results) > 0:
        ps_predictions = torch.argmax(ps_results, 1)
    else:
        ps_predictions = []
    if torch.numel(ng_results) > 0:
        ng_predictions = torch.argmax(ng_results, 1)
    else:
        ng_predictions = []
    return ps_predictions, ng_predictions

def _decode_pure_cls(
        tl_heat, br_heat, tl_regr, br_regr, cls, offset,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    # print(tl_scores)
    tl_regr_ = _tranpose_and_gather_feat(tl_regr, tl_inds)
    br_regr_ = _tranpose_and_gather_feat(br_regr, br_inds)


    tl_scores_ = tl_scores.view(1, batch, K)
    tl_clses_ = tl_clses.view(1, batch, K)
    tl_xs_ = tl_xs.view(1, batch, K)
    # print('_________________')
    # print(tl_xs_[0, 0])
    tl_ys_ = tl_ys.view(1, batch, K)
    tl_regr_ = tl_regr_.view(1, batch, K, 2)
    tl_xs_ += tl_regr_[:, :, :, 0]
    # print(tl_xs_[0, 0])
    tl_ys_ += tl_regr_[:, :, :, 1]
    br_scores_ = br_scores.view(1, batch, K)
    br_clses_ = br_clses.view(1, batch, K)
    br_xs_ = br_xs.view(1, batch, K)
    br_ys_ = br_ys.view(1, batch, K)
    br_regr_ = br_regr_.view(1, batch, K, 2)
    br_xs_ += br_regr_[:, :, :, 0]
    br_ys_ += br_regr_[:, :, :, 1]
    detections_tl = torch.cat([tl_scores_, tl_clses_.float(), tl_xs_, tl_ys_], dim=0)
    detections_br = torch.cat([br_scores_, br_clses_.float(), br_xs_, br_ys_], dim=0)
    print(cls)
    cls = torch.squeeze(torch.argmax(cls, 1))
    offset = torch.squeeze(offset)
    return detections_tl, detections_br, cls, offset

def _decode_pure_line(
        key_heat, hybrid_heat, key_tag, key_regr,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = key_heat.size()

    key_heat = torch.sigmoid(key_heat)
    hybrid_heat = torch.sigmoid(hybrid_heat)

    # perform nms on heatmaps
    key_heat = _nms(key_heat, kernel=kernel)
    hybrid_heat = _nms(hybrid_heat, kernel=kernel)

    key_scores, key_inds, key_clses, key_ys, key_xs = _topk(key_heat, K=K)
    hybrid_scores, hybrid_inds, hybrid_clses, hybrid_ys, hybrid_xs = _topk(hybrid_heat, K=K)
    # print(key_scores)
    key_regr_ = _tranpose_and_gather_feat(key_regr, key_inds)
    hybrid_regr_ = _tranpose_and_gather_feat(key_regr, hybrid_inds)
    key_tag_ = _tranpose_and_gather_feat(key_tag, key_inds)
    hybrid_tag_ = _tranpose_and_gather_feat(key_tag, hybrid_inds)

    key_tag_ = key_tag_.view(1, batch, K)
    hybrid_tag_ = hybrid_tag_.view(1, batch, K)
    key_scores_ = key_scores.view(1, batch, K)
    key_clses_ = key_clses.view(1, batch, K)
    key_xs_ = key_xs.view(1, batch, K)
    # print('_________________')
    # print(key_xs_[0, 0])
    key_ys_ = key_ys.view(1, batch, K)
    key_regr_ = key_regr_.view(1, batch, K, 2)
    key_xs_ += key_regr_[:, :, :, 0]
    # print(key_xs_[0, 0])
    key_ys_ += key_regr_[:, :, :, 1]
    hybrid_scores_ = hybrid_scores.view(1, batch, K)
    hybrid_clses_ = hybrid_clses.view(1, batch, K)
    hybrid_xs_ = hybrid_xs.view(1, batch, K)
    hybrid_ys_ = hybrid_ys.view(1, batch, K)
    hybrid_regr_ = hybrid_regr_.view(1, batch, K, 2)
    hybrid_xs_ += hybrid_regr_[:, :, :, 0]
    hybrid_ys_ += hybrid_regr_[:, :, :, 1]
    detections_key = torch.cat([key_scores_, key_tag_, key_clses_.float(), key_xs_, key_ys_], dim=0)
    detections_hybrid = torch.cat([hybrid_scores_, hybrid_tag_, hybrid_clses_.float(), hybrid_xs_, hybrid_ys_], dim=0)

    return detections_key, detections_hybrid

def _decode_pure_pie(
        center_heat, key_heat, center_regr, key_regr,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = center_heat.size()

    center_heat = torch.sigmoid(center_heat)
    key_heat = torch.sigmoid(key_heat)

    # perform nms on heatmaps
    center_heat = _nms(center_heat, kernel=kernel)
    key_heat = _nms(key_heat, kernel=kernel)

    center_scores, center_inds, center_clses, center_ys, center_xs = _topk(center_heat, K=K)
    key_scores, key_inds, key_clses, key_ys, key_xs = _topk(key_heat, K=K)

    center_regr_ = _tranpose_and_gather_feat(center_regr, center_inds)
    key_regr_ = _tranpose_and_gather_feat(key_regr, key_inds)

    center_scores_ = center_scores.view(1, batch, K)
    center_clses_ = center_clses.view(1, batch, K)
    center_xs_ = center_xs.view(1, batch, K)
    # print('_________________')
    # print(center_xs_[0, 0])
    center_ys_ = center_ys.view(1, batch, K)
    center_regr_ = center_regr_.view(1, batch, K, 2)
    center_xs_ += center_regr_[:, :, :, 0]
    # print(center_xs_[0, 0])
    center_ys_ += center_regr_[:, :, :, 1]
    key_scores_ = key_scores.view(1, batch, K)
    key_clses_ = key_clses.view(1, batch, K)
    key_xs_ = key_xs.view(1, batch, K)
    key_ys_ = key_ys.view(1, batch, K)
    key_regr_ = key_regr_.view(1, batch, K, 2)
    key_xs_ += key_regr_[:, :, :, 0]
    key_ys_ += key_regr_[:, :, :, 1]
    detections_center = torch.cat([center_scores_, center_clses_.float(), center_xs_, center_ys_], dim=0)
    detections_key = torch.cat([key_scores_, key_clses_.float(), key_xs_, key_ys_], dim=0)

    return detections_center, detections_key


def _decode_gt(
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, tl_gt, br_gt,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()
    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _getGT(tl_heat, tl_gt, K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _getGT(br_heat, br_gt, K)

    tl_tag_ = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag_ = tl_tag_.view(batch, K)
    br_tag_ = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag_ = br_tag_.view(batch, K)
    tl_regr_ = _tranpose_and_gather_feat(tl_regr, tl_inds)
    br_regr_ = _tranpose_and_gather_feat(br_regr, br_inds)

    tl_scores_ = tl_scores.view(1, batch, K)
    tl_tag_ = tl_tag_.view(1, batch, K)
    tl_clses_ = tl_clses.view(1, batch, K)
    tl_xs_ = tl_xs.view(1, batch, K)
    # print('_________________')
    # print(tl_xs_[0, 0])
    tl_ys_ = tl_ys.view(1, batch, K)
    tl_regr_ = tl_regr_.view(1, batch, K, 2)
    tl_xs_ += tl_regr_[:, :, :, 0]
    # print(tl_xs_[0, 0])
    tl_ys_ += tl_regr_[:, :, :, 1]
    br_scores_ = br_scores.view(1, batch, K)
    br_tag_ = br_tag_.view(1, batch, K)
    br_clses_ = br_clses.view(1, batch, K)
    br_xs_ = br_xs.view(1, batch, K)
    br_ys_ = br_ys.view(1, batch, K)
    br_regr_ = br_regr_.view(1, batch, K, 2)
    br_xs_ += br_regr_[:, :, :, 0]
    br_ys_ += br_regr_[:, :, :, 1]
    detections_tl = torch.cat([tl_scores_, tl_tag_, tl_clses_.float(), tl_xs_, tl_ys_], dim=0)
    detections_br = torch.cat([br_scores_, br_tag_, br_clses_.float(), br_xs_, br_ys_], dim=0)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    # print(tl_xs[0, :, 0])
    '''
    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
    '''
    # print(tl_xs[0, :, 0])
    # print('_________________')
    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections, detections_tl, detections_br


def _neg_loss(preds, gt, lamda, lamdb):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], lamda)
    loss = 0
    for pred in preds:
        # print(pred.shape)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        # print(pos_pred)
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, lamdb)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, lamdb) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


def _ae_loss(tag0, tag1, mask):
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _ae_line_loss(tag_full, mask_full):
    # mask_full [batch, Max_group, Max_len]
    # tag_full  [batch, Max_group, Max_len]
    pull = 0
    push = 0
    tag_full = torch.squeeze(tag_full)
    tag_full[1-mask_full] = 0
    num = mask_full.sum(dim=2, keepdim=True).float()
    tag_avg = tag_full.sum(dim=2, keepdim=True) / num
    pull = torch.pow(tag_full - tag_avg, 2) / (num + 1e-4)
    pull = pull[mask_full].sum()

    tag_avg = torch.squeeze(tag_avg)
    mask = mask_full.sum(dim=2)
    mask = mask.gt(1)
    num = mask.sum(dim=1, keepdim=True).float()
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)

    dist = tag_avg.unsqueeze(1) - tag_avg.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _ae_line_single(tags):
    pull = 0
    tag_means = []
    for tag in tags:
        num = tag.shape[0]
        tag_mean = tag.mean()
        tag = torch.pow(tag - tag_mean, 2) / (num + 1e-4)
        tag = tag.sum()
        pull += tag
        tag_means.append(tag_mean)
    tag_means = torch.cat(tag_means, 0)
    num = len(tags)
    num2 = (num - 1) * num
    dist = tag_means.unsqueeze(0) - tag_means.unsqueeze(1)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    push = dist.sum()
    return pull, push


def _regr_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _offset_loss(offset, gt_offset):
    offset_loss = nn.functional.smooth_l1_loss(offset, gt_offset, size_average=True)
    return offset_loss
