import os
import cv2
import json
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

def _clip_detections(image, ps_detections, ng_detections):
    height, width = image.shape[0:2]
    if len(ps_detections) > 0:
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

def kp_decode(nnet, inputs, K, ae_threshold=0.5, kernel=3):
    with torch.no_grad():
            detections, time_backbone, time_psn = nnet.test(inputs, ae_threshold=ae_threshold, K=K, kernel=kernel)
            #print(detections)
            ps_predictions = detections[0][inputs[5].squeeze()]
            ng_predictions = detections[1][inputs[6].squeeze()]
            ps_predictions = ps_predictions.data.cpu().numpy()
            ng_predictions = ng_predictions.data.cpu().numpy()
            return ps_predictions, ng_predictions, True

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

def kp_detection(db, nnet, result_dir, debug=False, decode_func=kp_decode):
    point_json_tl = os.path.join(result_dir, "points_tl.json")
    point_json_br = os.path.join(result_dir, "points_br.json")
    image_info_path = os.path.join(result_dir, "link_pre.json")
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    db_inds = db.db_inds
    num_images = db_inds.size

    K = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    nms_kernel = db.configs["nms_kernel"]

    scales = db.configs["test_scales"]
    weight_exp = db.configs["weight_exp"]
    merge_bbox = db.configs["merge_bbox"]
    categories = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1,
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]
    time_backbones = 0
    time_psns = 0
    max_tag_len = 32
    num_feature = 8
    total_acc = 0
    num_none = 0
    if True:
        image_info = {}
        for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
            db_ind = db_inds[ind]
            image_id   = db.image_ids(db_ind)
            image_file = db.image_file(db_ind)
            #print(image_file)
            image      = cv2.imread(image_file)
            (ps_detections, ng_detections) = db.detections(db_ind)
            if ps_detections is None or ng_detections is None:
                total_acc += 0
                num_none += 1
                continue
            ps_detections, ng_detections = _clip_detections(image, ps_detections, ng_detections)
            ps_detections = ps_detections[0:max_tag_len]
            ng_detections = ng_detections[0:max_tag_len]
            height, width = image.shape[0:2]

            scale = 1.0
            new_height = int(height * scale)
            new_width  = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width  = new_width  | 127
            images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios  = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes   = np.zeros((1, 2), dtype=np.float32)
            ps_tags = np.zeros((1, max_tag_len * 8 * 4), dtype=np.int64)
            ng_tags = np.zeros((1, max_tag_len * 8 * 4), dtype=np.int64)
            ps_weights = np.zeros((1, max_tag_len * 8 * 4), dtype=np.float32)
            ng_weights = np.zeros((1, max_tag_len * 8 * 4), dtype=np.float32)
            tag_masks_ps = np.zeros((1, max_tag_len), dtype=np.uint8)
            tag_masks_ng = np.zeros((1, max_tag_len), dtype=np.uint8)
            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])
            cv2.imwrite('test.png', resized_image)
            resized_image = resized_image / 255.
            images[0] = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0] = [inp_height, inp_width]
            ratios[0] = [height_ratio, width_ratio]
            if len(ps_detections) > 0:
                _rescale_points(ps_detections, ratios, borders, sizes)
            if len(ng_detections) > 0:
                _rescale_points(ng_detections, ratios, borders, sizes)
            #normalize_(resized_image, db.mean, db.std)


            tag_ind = 0
            b_ind = 0
            for k in range(len(ps_detections)):
                sp = ps_detections[k, 0]
                ep = ps_detections[k, 1]
                p_points, p_weights = _get_sample_point(sp, ep, num_feature)
                for kth in range(len(p_points)):
                    p_point = p_points[kth]
                    p_weight = p_weights[kth]
                    for sth in range(4):
                        ps_tags[b_ind, tag_ind + sth] = p_point[sth][1] * out_width + p_point[sth][0]
                        ps_weights[b_ind, tag_ind + sth] = p_weight[sth]
                    tag_ind += 4
                tag_masks_ps[b_ind, k] = 1
            tag_ind = 0
            for k in range(len(ng_detections)):
                sp = ng_detections[k, 0]
                ep = ng_detections[k, 1]
                n_points, n_weights = _get_sample_point(sp, ep, num_feature)
                for kth in range(len(n_points)):
                    n_point = n_points[kth]
                    n_weight = n_weights[kth]
                    for sth in range(4):
                        ng_tags[b_ind, tag_ind + sth] = n_point[sth][1] * out_width + n_point[sth][0]
                        ng_weights[b_ind, tag_ind + sth] = n_weight[sth]
                    tag_ind += 4
                tag_masks_ng[b_ind, k] = 1
            ps_tags = np.clip(ps_tags, 0, (out_width-1) * (out_height-1))
            ng_tags = np.clip(ng_tags, 0, (out_width-1) * (out_height-1))
            images = torch.from_numpy(images)
            ps_tags = torch.from_numpy(ps_tags)
            ng_tags = torch.from_numpy(ng_tags)
            ps_weights = torch.from_numpy(ps_weights)
            ng_weights = torch.from_numpy(ng_weights)
            tag_masks_ps = torch.from_numpy(tag_masks_ps)
            tag_masks_ng = torch.from_numpy(tag_masks_ng)
            ps_predictions, ng_predictions, flag = decode_func(nnet, [images, ps_tags, ng_tags, ps_weights, ng_weights, tag_masks_ps, tag_masks_ng], K, ae_threshold=ae_threshold, kernel=nms_kernel)
            if len(ps_detections) > 0:
                num_correct_ps = (ps_predictions == np.zeros_like(ps_predictions)).mean()
            else:
                num_correct_ps = 1
            if len(ng_detections) > 0:
                num_correct_ng = (ng_predictions == np.ones_like(ng_predictions)).mean()
            else:
                num_correct_ng = 1
            total_acc += (num_correct_ps+num_correct_ng)/2
            print((num_correct_ps+num_correct_ng)/2)
            ps_predictions = ps_predictions.tolist()
            ng_predictions = ng_predictions.tolist()
            image_info[image_id] = {'ps_predictions': ps_predictions, 'ng_predictions': ng_predictions}

            #print(0)
            #print(dets_tl)
            if not flag:
                print("error when try to test %s" %image_file)
                continue

        with open(image_info_path, "w") as f:
            json.dump(image_info, f)
        print('Final Avg Acc: %f' %(total_acc/(num_images-num_none)))
    '''
    image_ids = [db.image_ids(ind) for ind in db_inds]
    with open(result_json, "r") as f:
        result_json = json.load(f)
    for cls_type in range(1, categories+1):
        db.evaluate(result_json, [cls_type], image_ids)
    '''
    return 0

def testing(db, nnet, result_dir, debug=False):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug)
