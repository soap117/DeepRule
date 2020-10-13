import os
import cv2
import json
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

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
    try:
        detections_tl, detections_br = nnet.test([images], ae_threshold=ae_threshold, K=K, kernel=kernel)
        detections_tl = detections_tl.data.cpu().numpy()
        detections_br = detections_br.data.cpu().numpy()
        return detections_tl, detections_br, True
    except Exception as e:
        print("memory out skip")
        torch.cuda.empty_cache()
        return None, None, False

def kp_detection(db, nnet, result_dir, debug=False, decode_func=kp_decode):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval":
        db_inds = db.db_inds[:100] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]
    num_images = db_inds.size

    K             = db.configs["top_k"]
    ae_threshold  = db.configs["ae_threshold"]
    nms_kernel    = db.configs["nms_kernel"]
    
    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    max_height = 600
    max_width = 1000
    detections = []
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[ind]

        image_file = db.image_file(db_ind)
        ori_image = cv2.imread(image_file)
        ori_height, ori_width = ori_image.shape[0:2]
        print("image nmae: %s, width: %d, height: %d" %(image_file, ori_width, ori_height))
        height = min(max_height, ori_height)
        width = min(max_width, ori_width)
        input_image = cv2.resize(ori_image, (width, height))

        inp_height = max_height | 127
        inp_width = max_width | 127
        input_image_full = cv2.resize(ori_image, (inp_width, inp_height))
        images = np.zeros((2, 3, inp_height, inp_width), dtype=np.float32)

        input_image = input_image / 255.
        normalize_(input_image, db.mean, db.std)
        input_image_full = input_image_full / 255.
        normalize_(input_image_full, db.mean, db.std)
        out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4

        images[0, :, 0:height, 0:width] = input_image.transpose((2, 0, 1))
        images[1] = input_image_full.transpose((2, 0, 1))
        images = torch.from_numpy(images)
        detections_tl, detections_br, flag = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel)


        ratio_height = inp_height / out_height * ori_height / height
        ratio_width = inp_width / out_width * ori_width / width
        detections_tl[3, 0] *= ratio_width
        detections_tl[4, 0] *= ratio_height
        detections_br[3, 0] *= ratio_width
        detections_br[4, 0] *= ratio_height

        ratio_height = inp_height / out_height * ori_height / inp_height
        ratio_width = inp_width / out_width * ori_width / inp_width
        detections_tl[3, 1] *= ratio_width
        detections_tl[4, 1] *= ratio_height
        detections_br[3, 1] *= ratio_width
        detections_br[4, 1] *= ratio_height


        if flag:
            detections.append([detections_tl, detections_br])


    result_pickle = os.path.join(result_dir, "results_points.pickle")
    with open(result_pickle, "wb") as f:
        pickle.dump(detections, f)

    return 0

def testing(db, nnet, result_dir, debug=False):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug)
