import cv2
import math
import numpy as np
import torch
import random
import string

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

def random_crop_line(image, detections, random_scales, view_size, border=64):
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
    cropped_detections = detections.copy()
    cropped_detections[:, 0:cropped_detections.shape[1]:2] -= x0
    cropped_detections[:, 1:cropped_detections.shape[1]:2] -= y0
    cropped_detections[:, 0:cropped_detections.shape[1]:2] += cropped_ctx - left_w
    cropped_detections[:, 1:cropped_detections.shape[1]:2] += cropped_cty - top_h

    return cropped_image, cropped_detections, scale

def _resize_image(image, detections, size):
    detections = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    detections[:, 0:detections.shape[1]:2] *= width_ratio
    detections[:, 1:detections.shape[1]:2] *= height_ratio
    return image, detections


def _clip_detections(image, detections):
    detections = detections.copy()
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
    detections[:, 0:detections.shape[1]:2] = np.clip(detections[:, 0:detections.shape[1]:2], 0, width)
    detections[:, 1:detections.shape[1]:2] = np.clip(detections[:, 1:detections.shape[1]:2], 0, height)
    return detections


def kp_detection(db, k_ind, data_aug, debug):
    data_rng = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories = db.configs["categories"]
    input_size = db.configs["input_size"]
    output_size = db.configs["output_sizes"][0]

    border = db.configs["border"]
    lighting = db.configs["lighting"]
    rand_crop = db.configs["rand_crop"]
    rand_color = db.configs["rand_color"]
    rand_scales = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou = db.configs["gaussian_iou"]
    gaussian_rad = db.configs["gaussian_radius"]

    max_tag_len = 256
    max_tag_len_group = 128
    max_group_len = 16
    # allocating memory
    images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    key_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    hybrid_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    key_regrs = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    key_tags = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    key_tags_grouped = np.zeros((batch_size, max_group_len, max_tag_len_group), dtype=np.int64)
    tag_masks = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_masks_grouped = np.zeros((batch_size, max_group_len, max_tag_len_group), dtype=np.uint8)
    hybrid_masks_grouped = np.zeros((batch_size, max_group_len, max_tag_len_group), dtype=np.uint8)
    tag_lens = np.zeros((batch_size,), dtype=np.int32)
    tag_group_lens = np.zeros((batch_size,), dtype=np.int32)

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
            temp = db.detections(db_ind)
            if temp == None:
                flag = False
        ori_size = image.shape
        #print(temp)
        (detections, categories) = temp
        detections = detections[0:max_group_len]
        categories = categories[0:max_group_len]
        # cropping an image randomly
        if not debug and rand_crop:
            image, detections, scale = random_crop_line(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)
        # print("Image_size")
        # print(image.shape)
        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

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

        for ind, (detection, category) in enumerate(zip(detections, categories)):
            category = 0
            # print("Category: %d" %category)
            # print("Detections: %d" % len(detections))
            fdetection = detection.copy()
            fdetection[0:len(fdetection):2] = detection[0:len(detection):2] * width_ratio
            fdetection[1:len(fdetection):2] = detection[1:len(detection):2] * height_ratio
            detection = fdetection.astype(np.int32)

            if gaussian_bump:
                width = ori_size[1] / 50 / 4 / scale
                height = ori_size[0] / 50 / 4 / scale

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                for k in range(int(len(detection) / 2)):
                    if not (detection[2*k] == 0 or detection[2*k+1] == 0 or detection[2*k] >= (output_size[1]-1e-2) or detection[2*k+1] >= (output_size[0]-1e-2)):
                        if key_heatmaps[b_ind, category, detection[2 * k + 1], detection[2 * k]] < 0.85:
                            draw_gaussian(key_heatmaps[b_ind, category], [detection[2 * k], detection[2 * k + 1]], radius)
                        else:
                            draw_gaussian(key_heatmaps[b_ind, category], [detection[2 * k], detection[2 * k + 1]], radius)
                            draw_gaussian(hybrid_heatmaps[b_ind, category], [detection[2 * k], detection[2 * k + 1]],
                                          radius)
            else:
                key_heatmaps[b_ind, category, detection[2 * k + 1], detection[2 * k]] = 1

            for k in range(int(len(detection) / 2)):
                if not (detection[2 * k] == 0 or detection[2 * k + 1] == 0 or detection[2*k] >= (output_size[1]-1e-2) or detection[2*k+1] >= (output_size[0]-1e-2)):
                    if tag_lens[b_ind] >= max_tag_len - 1 or k > max_tag_len_group - 1:
                        print("Too many targets, skip!")
                        print(tag_lens[b_ind])
                        print(image_file)
                        break
                    tag_ind = tag_lens[b_ind]
                    key_regrs[b_ind, tag_ind, :] = [fdetection[2 * k] - detection[2 * k],
                                                    fdetection[2 * k + 1] - detection[2 * k + 1]]
                    key_tags[b_ind, tag_ind] = detection[2 * k + 1] * output_size[1] + detection[2 * k]
                    key_tags_grouped[b_ind, ind, k] = detection[2 * k + 1] * output_size[1] + detection[2 * k]
                    tag_lens[b_ind] += 1
                    if hybrid_heatmaps[b_ind, category, detection[2 * k + 1], detection[2 * k]] < 0.85:
                        tag_masks_grouped[b_ind, ind, k] = 1
                    # print("Pre_tag_ing:%d" %tag_ind)
            tag_len = tag_lens[b_ind]
            tag_group_lens[b_ind] += 1
            tag_masks[b_ind, :tag_len] = 1

    tag_masks_grouped = tag_masks_grouped * (1 - hybrid_masks_grouped)
    images = torch.from_numpy(images)
    key_heatmaps = torch.from_numpy(key_heatmaps)
    key_regrs = torch.from_numpy(key_regrs)
    key_tags = torch.from_numpy(key_tags)
    tag_masks = torch.from_numpy(tag_masks)
    key_tags_grouped = torch.from_numpy(key_tags_grouped)
    tag_group_lens = torch.from_numpy(tag_group_lens)
    hybrid_heatmaps = torch.from_numpy(hybrid_heatmaps)
    tag_masks_grouped = torch.from_numpy(tag_masks_grouped)
    return {
               "xs": [images, key_tags, key_tags_grouped, tag_group_lens],
               "ys": [key_heatmaps, hybrid_heatmaps, tag_masks, tag_masks_grouped, key_regrs]
           }, k_ind


def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
