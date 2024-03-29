import cv2
import math
import numpy as np
import torch
import random
import string

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop_pie, draw_gaussian, gaussian_radius

def _full_image_crop(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size   = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:, 0:4:2] += border[2]
    detections[:, 1:4:2] += border[0]
    return image, detections

def _resize_image(image, detections, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:6:2] *= width_ratio
    detections[:, 1:6:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:6:2] = np.clip(detections[:, 0:6:2], 0, width - 1)
    detections[:, 1:6:2] = np.clip(detections[:, 1:6:2], 0, height - 1)
    return detections

def kp_detection(db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    max_tag_len = 128

    # allocating memory
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    center_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    key_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    center_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    key_regrs_tl = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    key_regrs_br = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    center_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    key_tags_tl     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    key_tags_br = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.bool_)
    tag_lens    = np.zeros((batch_size, ), dtype=np.int32)

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

        # reading detections
        detections = db.detections(db_ind)

        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop_pie(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)
        #print("Image_size")
        #print(image.shape)
        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]


        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            #normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        for ind, detection in enumerate(detections):

            category = int(detection[-1]) - 1
            if category == 2:
                category = 0
            #print("Category: %d" %category)
            #print("Detections: %d" % len(detections))
            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            xce, yce = detection[4], detection[5]

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)
            fxce = (xce * width_ratio)
            fyce = (yce * height_ratio)
            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)
            xce = int(fxce)
            yce = int(fyce)

            if gaussian_bump:
                width = math.sqrt(math.pow(xce-xtl, 2)+math.pow(yce-ytl, 2))
                height = math.sqrt(math.pow(xce-xbr, 2)+math.pow(yce-ybr, 2))

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad

                draw_gaussian(center_heatmaps[b_ind, category], [xce, yce], radius)
                draw_gaussian(key_heatmaps[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(key_heatmaps[b_ind, category], [xbr, ybr], radius)
            else:
                center_heatmaps[b_ind, category, yce, xce] = 1
                key_heatmaps[b_ind, category, ytl, xtl] = 1
                key_heatmaps[b_ind, category, ybr, xbr] = 1

            tag_ind = tag_lens[b_ind]
            center_regrs[b_ind, tag_ind, :] = [fxce - xce, fyce - yce]
            key_regrs_tl[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
            key_regrs_br[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
            center_tags[b_ind, tag_ind] = yce * output_size[1] + xce
            key_tags_tl[b_ind, tag_ind] = ytl * output_size[1] + xtl
            key_tags_br[b_ind, tag_ind] = ybr * output_size[1] + xbr
            tag_lens[b_ind] += 1
            if tag_lens[b_ind] >= max_tag_len-1:
                print("Too many targets, skip!")
                print(tag_lens[b_ind])
                print(image_file)
                break
            #print("Pre_tag_ing:%d" %tag_ind)
    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    images      = torch.from_numpy(images)
    key_heatmaps = torch.from_numpy(key_heatmaps)
    center_heatmaps = torch.from_numpy(center_heatmaps)
    center_regrs = torch.from_numpy(center_regrs)
    key_regrs_tl = torch.from_numpy(key_regrs_tl)
    key_regrs_br = torch.from_numpy(key_regrs_br)
    center_tags = torch.from_numpy(center_tags)
    key_tags_tl = torch.from_numpy(key_tags_tl)
    key_tags_br = torch.from_numpy(key_tags_br)
    tag_masks   = torch.from_numpy(tag_masks)

    return {
        "xs": [images, center_tags, key_tags_tl, key_tags_br],
        "ys": [center_heatmaps, key_heatmaps, tag_masks, center_regrs, key_regrs_tl, key_regrs_br]
    }, k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
