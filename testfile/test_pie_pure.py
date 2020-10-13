import os
import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
import external.nms as nms

def _rescale_points(dets, ratios, borders, sizes):
    xs, ys = dets[:, :, 2], dets[:, :, 3]
    xs    /= ratios[0, 1]
    ys    /= ratios[0, 0]
    xs    -= borders[0, 2]
    ys    -= borders[0, 0]
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

def kp_decode(nnet, images, K, ae_threshold=0.5, kernel=3):
    with torch.no_grad():
            detections_tl_detections_br, time_backbone, time_psn = nnet.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel)
            detections_tl = detections_tl_detections_br[0]
            detections_br = detections_tl_detections_br[1]
            detections_tl = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
            detections_br = detections_br.data.cpu().numpy().transpose((2, 1, 0))
            return detections_tl, detections_br, time_backbone, time_psn, True


def kp_detection(db, nnet, result_dir, debug=False, decode_func=kp_decode):
    point_json_tl = os.path.join(result_dir, "points_center.json")
    point_json_br = os.path.join(result_dir, "points_key.json")
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
    if True:

        top_points_tl = {}
        top_points_br = {}
        for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
            db_ind = db_inds[ind]
            image_id   = db.image_ids(db_ind)
            image_file = db.image_file(db_ind)
            #print(image_file)
            image      = cv2.imread(image_file)

            height, width = image.shape[0:2]

            detections_point_tl = []
            detections_point_br = []
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

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            #normalize_(resized_image, db.mean, db.std)

            images[0]  = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]

            images = torch.from_numpy(images)
            dets_tl, dets_br, time_backbone, time_psn, flag = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)
            time_backbones += time_backbone
            time_psns += time_psn
            #print('b time:%f' % (time_backbones / float(ind + 1)))
            #print('p time:%f' % (time_psns / float(ind + 1)))
            #print(0)
            #print(dets_tl)
            if not flag:
                print("error when try to test %s" %image_file)
                continue
            #print(dets_tl.shape)
            _rescale_points(dets_tl, ratios, borders, sizes)
            _rescale_points(dets_br, ratios, borders, sizes)
            #print(dets_tl)
            detections_point_tl.append(dets_tl)
            detections_point_br.append(dets_br)
            detections_point_tl = np.concatenate(detections_point_tl, axis=1)
            detections_point_br = np.concatenate(detections_point_br, axis=1)
            #print(detections_point_tl[:, 0, 0])
            #print('1')
            #print(detections_point.shape)

            classes_p_tl = detections_point_tl[:, 0, 1]
            classes_p_br = detections_point_br[:, 0, 1]
            #print('2')
            #print(classes_p.shape)

            # reject detections with negative scores

            keep_inds_p = (detections_point_tl[:, 0, 0] > 0)
            detections_point_tl = detections_point_tl[keep_inds_p, 0]
            classes_p_tl = classes_p_tl[keep_inds_p]

            keep_inds_p = (detections_point_br[:, 0, 0] > 0)
            detections_point_br = detections_point_br[keep_inds_p, 0]
            classes_p_br = classes_p_br[keep_inds_p]

            #print('3')
            #print(detections_point.shape)

            top_points_tl[image_id] = {}
            top_points_br[image_id] = {}
            for j in range(categories):

                keep_inds_p = (classes_p_tl == j)
                top_points_tl[image_id][j + 1] = detections_point_tl[keep_inds_p].astype(np.float32)
                keep_inds_p = (classes_p_br == j)
                top_points_br[image_id][j + 1] = detections_point_br[keep_inds_p].astype(np.float32)
                #print(top_points[image_id][j + 1][0])


            scores = np.hstack([
                top_points_tl[image_id][j][:, 0]
                for j in range(1, categories + 1)
            ])
            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, categories + 1):
                    keep_inds = (top_points_tl[image_id][j][:, 0] >= thresh)
                    top_points_tl[image_id][j] = top_points_tl[image_id][j][keep_inds]

            scores = np.hstack([
                top_points_br[image_id][j][:, 0]
                for j in range(1, categories + 1)
            ])
            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, categories + 1):
                    keep_inds = (top_points_br[image_id][j][:, 0] >= thresh)
                    top_points_br[image_id][j] = top_points_br[image_id][j][keep_inds]

            if debug:
                image_file = db.image_file(db_ind)
                image      = cv2.imread(image_file)

        #print(top_points_tl)
        detections_point_tl = db.convert_to_coco_points_pure(top_points_tl)
        detections_point_br = db.convert_to_coco_points_pure(top_points_br)
        with open(point_json_tl, "w") as f:
            json.dump(detections_point_tl, f)
        with open(point_json_br, "w") as f:
            json.dump(detections_point_br, f)
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
