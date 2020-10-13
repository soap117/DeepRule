import os
import cv2
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import system_configs
import math
import external.nms as nms

def _rescale_points(dets, ratios, borders, sizes):
    xs, ys = dets[:, :, 0], dets[:, :, 1]
    xs    += borders[0, 2]
    ys    += borders[0, 0]
    xs *= ratios[0, 1]
    ys *= ratios[0, 0]
    np.clip(xs, 0, sizes[0, 1], out=xs)
    np.clip(ys, 0, sizes[0, 0], out=ys)

def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

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

def _clip_detections(image, detections):
    height, width = image.shape[0:2]
    if len(detections) > 0:
        keep_inds = ((detections[:, 0, 0] > 0) & (detections[:, 0, 0] < width) & (detections[:, 0, 1] > 0) & (detections[:, 0, 1] < height)
                     &(detections[:, 1, 0] > 0) & (detections[:, 1, 0] < width) & (detections[:, 1, 1] > 0) & (detections[:, 1, 1] < height))
        detections = detections[keep_inds]
    return detections

def kp_decode(nnet, inputs, K, ae_threshold=0.5, kernel=3):
    with torch.no_grad():
            detections, time_backbone, time_psn = nnet.test(inputs, ae_threshold=ae_threshold, K=K, kernel=kernel)
            #print(detections)
            predictions = detections[inputs[3].squeeze()]
            predictions = predictions.data.cpu().numpy()
            return predictions, True

def crop_image(image, center, size):
    cty, ctx            = center
    height, width       = size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset

def kp_detection(image, db, quiry, nnet, debug=False, decode_func=kp_decode, cuda_id=0):


    K = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    nms_kernel = db.configs["nms_kernel"]
    max_tag_len = 400
    num_feature = 8
    total_acc = 0
    num_none = 0
    detections = np.array(quiry)
    if True:
        detections = _clip_detections(image, detections)
        ori_detections = copy.deepcopy(detections)
        height, width = image.shape[0:2]

        scale = 1.0
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width = new_width | 127
        images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes = np.zeros((1, 2), dtype=np.float32)
        tags = np.zeros((1, max_tag_len * 8 * 4), dtype=np.int64)
        weights = np.zeros((1, max_tag_len * 8 * 4), dtype=np.float32)
        tag_masks = np.zeros((1, max_tag_len), dtype=np.uint8)
        out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
        height_ratio = out_height / inp_height
        width_ratio = out_width / inp_width
        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])
        cv2.imwrite('test.png', resized_image)
        resized_image = resized_image / 255.
        images[0] = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0] = [inp_height, inp_width]
        ratios[0] = [height_ratio, width_ratio]
        if len(detections) > 0:
            _rescale_points(detections, ratios, borders, sizes)
        # normalize_(resized_image, db.mean, db.std)

        tag_ind = 0
        b_ind = 0
        print(len(detections))
        for k in range(len(detections)):
            sp = detections[k, 0]
            ep = detections[k, 1]
            p_points, p_weights = _get_sample_point(sp, ep, num_feature)
            for kth in range(len(p_points)):
                p_point = p_points[kth]
                p_weight = p_weights[kth]
                for sth in range(4):
                    tags[b_ind, tag_ind + sth] = p_point[sth][1] * out_width + p_point[sth][0]
                    weights[b_ind, tag_ind + sth] = p_weight[sth]
                tag_ind += 4
            tag_masks[b_ind, k] = 1
        tags = np.clip(tags, 0, (out_width - 1) * (out_height - 1))
        if torch.cuda.is_available():
            images = torch.from_numpy(images).cuda(cuda_id)
            tags = torch.from_numpy(tags).cuda(cuda_id)
            weights = torch.from_numpy(weights).cuda(cuda_id)
            tag_masks = torch.from_numpy(tag_masks).cuda(cuda_id)
        else:
            images = torch.from_numpy(images)
            tags = torch.from_numpy(tags)
            weights = torch.from_numpy(weights)
            tag_masks = torch.from_numpy(tag_masks)
        predictions, flag = decode_func(nnet, [images, tags, weights, tag_masks], K, ae_threshold=ae_threshold,
                                        kernel=nms_kernel)
        #print(predictions)
        predictions = predictions.tolist()
        pair2pre = {}
        ori_detections = ori_detections.tolist()
        for det, pre in zip(ori_detections, predictions):
            pair2pre[str(det)] = pre
        return pair2pre



def testing(image, db, quiry, nnet, debug=False, decode_func=kp_decode, cuda_id=0):
    return globals()[system_configs.sampling_function](image, db, quiry, nnet, debug=debug, cuda_id=cuda_id)
