import cv2
import math
import numpy as np
import torch
import random
import random

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import draw_gaussian, gaussian_radius, _get_border


def _full_image_crop(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:, 0:4:2] += border[2]
    detections[:, 1:4:2] += border[0]
    return image, detections

def random_crop_line(image, ps_detections, ng_detections, random_scales, view_size, border=64):
    view_height, view_width   = view_size
    image_height, image_width = image.shape[0:2]

    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    ps_cropped_detections = ps_detections.copy()
    ps_cropped_detections[:, :, 0] -= x0
    ps_cropped_detections[:, :, 1] -= y0
    ps_cropped_detections[:, :, 0] += cropped_ctx - left_w
    ps_cropped_detections[:, :, 1] += cropped_cty - top_h

    ng_cropped_detections = ng_detections.copy()
    if len(ng_detections) > 0:
        ng_cropped_detections[:, :, 0] -= x0
        ng_cropped_detections[:, :, 1] -= y0
        ng_cropped_detections[:, :, 0] += cropped_ctx - left_w
        ng_cropped_detections[:, :, 1] += cropped_cty - top_h
    return cropped_image, ps_cropped_detections, ng_cropped_detections, scale

def _resize_image(image, ps_detections, ng_detections, size):
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    ps_detections = ps_detections.copy()
    ps_detections[:, :, 0] *= width_ratio
    ps_detections[:, :, 1] *= height_ratio
    ng_detections = ng_detections.copy()
    if len(ng_detections) > 0:
        ng_detections[:, :, 0] *= width_ratio
        ng_detections[:, :, 1] *= height_ratio
    return image, ps_detections, ng_detections


def _clip_detections(image, ps_detections, ng_detections):
    height, width = image.shape[0:2]
    '''
    for ind in range(16):
        for k in range(int(len(detections[0])/2)):
            if detections[0, 2*k] >= width:
                detections[ind, 2 * k + 1] = (detections[ind, 2 * k + 1] - detections[ind, 2 * k - 1]) / (
                            detections[ind, 2 * k] - detections[ind, 2 * (k - 1)] + 1e-3) * (
                                                       width - 1 - detections[ind, 2 * (k - 1)]) + detections[ind, 2 * k - 1]
                detections[ind, 2 * k] = width - 1
    '''
    keep_inds = ((ps_detections[:, 0, 0] > 0) & (ps_detections[:, 0, 0] < width) & (ps_detections[:, 0, 1] > 0) & (ps_detections[:, 0, 1] < height)
                 &(ps_detections[:, 1, 0] > 0) & (ps_detections[:, 1, 0] < width) & (ps_detections[:, 1, 1] > 0) & (ps_detections[:, 1, 1] < height))
    ps_detections = ps_detections[keep_inds]
    if len(ng_detections) > 0:
        keep_inds = ((ng_detections[:, 0, 0] > 0) & (ng_detections[:, 0, 0] < width) & (ng_detections[:, 0, 1] > 0) & (
                    ng_detections[:, 0, 1] < height)
                     & (ng_detections[:, 1, 0] > 0) & (ng_detections[:, 1, 0] < width) & (ng_detections[:, 1, 1] > 0) & (
                                 ng_detections[:, 1, 1] < height))
        ng_detections = ng_detections[keep_inds]
    return ps_detections, ng_detections


def _get_sample_point(s, e, num_feature):
    dx = (e[0] - s[0]) / (num_feature-1)
    ke = (e[1] - s[1]) / (e[0]-s[0]+1e-4)
    points = []
    weights = []
    for k in range(num_feature):
        xm = dx*k+s[0]
        ym = (xm-s[0])*ke + s[1]
        xc = math.ceil(xm)
        yc = math.ceil(ym)
        xf = math.floor(xm)
        yf = math.floor(ym)
        points.append([[xf, yf],[xf, yc],[xc, yf],[xc, yc]])
        weights.append([(1-(xm-xf))*(1-(ym-yf)), (1-(xm-xf))*(1-(yc-ym)), (1-(xc-xm))*(1-(ym-yf)), (1-(xc-xm))*(1-(yc-ym))])
    return points, weights


def kp_detection(db, k_ind, data_aug, debug):
    data_rng = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories = db.configs["categories"]
    input_size = db.configs["input_size"]
    output_size = db.configs["output_sizes"][0]

    border = db.configs["border"]
    lighting = db.configs["lighting"] and data_aug
    rand_crop = db.configs["rand_crop"]
    rand_color = db.configs["rand_color"] and data_aug
    rand_scales = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou = db.configs["gaussian_iou"]
    gaussian_rad = db.configs["gaussian_radius"]

    max_tag_len = 16
    max_group_len = 16
    num_feature = 8
    # allocating memory
    images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    ps_tags = np.zeros((batch_size, max_tag_len * 8 * 4), dtype=np.int64)
    ng_tags = np.zeros((batch_size, max_tag_len * 8 * 4), dtype=np.int64)
    ps_weights = np.zeros((batch_size, max_tag_len * 8 * 4), dtype=np.float32)
    ng_weights = np.zeros((batch_size, max_tag_len * 8 * 4), dtype=np.float32)
    tag_masks_ps = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_masks_ng = np.zeros((batch_size, max_tag_len), dtype=np.uint8)

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()
        flag = False
        while not flag:
            db_ind = db.db_inds[k_ind]
            k_ind = (k_ind + 1) % db_size
            # reading image
            image_file = db.image_file(db_ind)
            # print(image_file)
            image = cv2.imread(image_file)
            if image.any() != None:
                flag = True
            (ps_detections, ng_detections) = db.detections(db_ind)
            if ps_detections is None:
                flag = False
                continue
            if len(ps_detections) < 1:
                flag = False
                continue
            ori_size = image.shape
            #print(temp)
            ps_detections = np.array(ps_detections)
            ng_detections = np.array(ng_detections)
            # cropping an image randomly
            if not debug and rand_crop:
                image, ps_detections, ng_detections, scale = random_crop_line(image, ps_detections, ng_detections, rand_scales, input_size, border=border)
            else:
                image, detections = _full_image_crop(image, detections)
            # print("Image_size")
            # print(image.shape)
            image, ps_detections, ng_detections = _resize_image(image, ps_detections, ng_detections, input_size)
            ps_detections, ng_detections = _clip_detections(image, ps_detections, ng_detections)
            if len(ps_detections) < 1:
                flag = False

        np.random.shuffle(ps_detections)
        np.random.shuffle(ng_detections)
        ps_detections = ps_detections[0:max_group_len]
        ng_detections = ng_detections[0:max_group_len]
        #cv2.imwrite('test.png', image)
        width_ratio = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            # normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        ps_fdetections = ps_detections.copy()
        ps_fdetections[:, :, 0] = ps_detections[:, :, 0] * width_ratio
        ps_fdetections[:, :, 1] = ps_detections[:, :, 1] * height_ratio
        ng_fdetections = ng_detections.copy()
        if len(ng_detections) > 0:
            ng_fdetections[:, :, 0] = ng_detections[:, :, 0] * width_ratio
            ng_fdetections[:, :, 1] = ng_detections[:, :, 1] * height_ratio
        tag_ind = 0
        for k in range(len(ps_detections)):
            sp = ps_fdetections[k, 0]
            ep = ps_fdetections[k, 1]
            p_points, p_weights = _get_sample_point(sp, ep, num_feature)
            for kth in range(len(p_points)):
                p_point = p_points[kth]
                p_weight = p_weights[kth]
                for sth in range(4):
                    ps_tags[b_ind, tag_ind+sth] = p_point[sth][1] * output_size[1] + p_point[sth][0]
                    ps_weights[b_ind, tag_ind+sth] = p_weight[sth]
                tag_ind += 4
            tag_masks_ps[b_ind, k] = 1
        tag_ind = 0
        for k in range(len(ng_detections)):
            sp = ng_fdetections[k, 0]
            ep = ng_fdetections[k, 1]
            n_points, n_weights = _get_sample_point(sp, ep, num_feature)
            for kth in range(len(n_points)):
                n_point = n_points[kth]
                n_weight = n_weights[kth]
                for sth in range(4):
                    ng_tags[b_ind, tag_ind+sth] = n_point[sth][1] * output_size[1] + n_point[sth][0]
                    ng_weights[b_ind, tag_ind+sth] = n_weight[sth]
                tag_ind += 4
            tag_masks_ng[b_ind, k] = 1
    ps_tags = np.clip(ps_tags, 0, 127 * 127)
    ng_tags = np.clip(ng_tags, 0, 127 * 127)
    images = torch.from_numpy(images)
    ps_tags = torch.from_numpy(ps_tags)
    ng_tags = torch.from_numpy(ng_tags)
    ps_weights = torch.from_numpy(ps_weights)
    ng_weights = torch.from_numpy(ng_weights)
    tag_masks_ps = torch.from_numpy(tag_masks_ps)
    tag_masks_ng = torch.from_numpy(tag_masks_ng)
    return {
               "xs": [images, ps_tags, ng_tags, ps_weights, ng_weights],
               "ys": [torch.zeros([batch_size, 16], dtype=torch.int64), torch.ones([batch_size, 16], dtype=torch.int64), tag_masks_ps, tag_masks_ng]
           }, k_ind


def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)


