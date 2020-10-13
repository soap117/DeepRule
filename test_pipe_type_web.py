#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse

import matplotlib
matplotlib.use("Agg")
import cv2
from tqdm import tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib
import json
from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBarRaw
from RuleGroup.LineQuiry import GroupQuiryRaw
from RuleGroup.LIneMatch import GroupLineRaw
from RuleGroup.Pie import GroupPie
import math
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
torch.backends.cudnn.benchmark = False
import requests
import time
import re
def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", dest="cfg_file", help="config file", default="CornerNetLine", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=50000, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument('--cache_path', dest="cache_path", type=str)
    parser.add_argument('--result_path', dest="result_path", type=str)
    parser.add_argument('--tar_data_path', dest="tar_data_path", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_dir", dest="data_dir", default="data/linedata(1028)", type=str)
    parser.add_argument("--image_dir", dest="image_dir", default="C:/work/linedata(1028)/line/images/test2019/f4b5dac780890c2ca9f43c3fe4cc991a_d3d3LmVwc2lsb24uaW5zZWUuZnIJMTk1LjEwMS4yNTEuMTM2.xls-3-0.png", type=str)
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
def load_net(testiter, cfg_name, data_dir, cache_dir, result_dir, cuda_id=0):
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["result_dir"] = result_dir
    configs["system"]["tar_data_dir"] = "Cls"
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["validation"]

    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    make_dirs([result_dir])

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet

def Pre_load_nets(type, id_cuda):
    methods = {}
    db_cls, nnet_cls = load_net(50000, "CornerNetCls", "data/clsdata(1031)", "data/clsdata(1031)/cache",
                                "data/clsdata(1031)/result")
    path = 'testfile.test_%s' % "CornerNetCls"
    testing_cls = importlib.import_module(path).testing
    methods['Cls'] = [db_cls, nnet_cls, testing_cls]
    if type == 0:
        db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", "data/bardata(1031)", "data/bardata(1031)/cache",
                                    "data/bardata(1031)/result", id_cuda)
        path = 'testfile.test_%s' % "CornerNetPureBar"
        testing_bar = importlib.import_module(path).testing
        methods['Bar'] = [db_bar, nnet_bar, testing_bar]
    if type == 2:
        db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", "data/piedata(1008)", "data/piedata(1008)/cache",
                                    "data/piedata(1008)/result", id_cuda)
        path = 'testfile.test_%s' % "CornerNetPurePie"
        testing_pie = importlib.import_module(path).testing
        methods['Pie'] = [db_pie, nnet_pie, testing_pie]
    if type == 1:
        db_line, nnet_line = load_net(50000, "CornerNetLine", "data/linedata(1028)", "data/linedata(1028)/cache",
                                      "data/linedata(1028)/result", id_cuda)
        path = 'testfile.test_%s' % "CornerNetLine"
        testing_line = importlib.import_module(path).testing
        methods['Line'] = [db_line, nnet_line, testing_line]
        db_line_cls, nnet_line_cls = load_net(20000, "CornerNetLineClsReal", "data/linedata(1028)",
                                              "data/linedata(1028)/cache",
                                              "data/linedata(1028)/result", id_cuda)
        path = 'testfile.test_%s' % "CornerNetLineCls"
        testing_line_cls = importlib.import_module(path).testing
        methods['LineCls'] = [db_line_cls, nnet_line_cls, testing_line_cls]
    return methods

def ocr_result(image_path):
    subscription_key = "ad143190288d40b79483aa0d5c532724"
    vision_base_url = "https://westus2.api.cognitive.microsoft.com/vision/v2.0/"
    ocr_url = vision_base_url + "read/core/asyncBatchAnalyze"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'language': 'eng', 'detectOrientation': 'true'}

    image = Image.open(image_path)
    enh_con = ImageEnhance.Contrast(image)
    contrast = 2.0
    image = enh_con.enhance(contrast)
    image = image.convert('L')
    # image = image.resize((800, 800))
    image.save('OCR_temp.png')
    image_data = open('OCR_temp.png', "rb").read()
    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    op_location = response.headers['Operation-Location']
    analysis = {}
    while "recognitionResults" not in analysis.keys():
        time.sleep(3)
        binary_content = requests.get(op_location, headers=headers, params=params).content
        analysis = json.loads(binary_content.decode('ascii'))
    line_infos = [region["lines"] for region in analysis["recognitionResults"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                if 'confidence' in word_info.keys():
                    if word_info['confidence'] == 'Low':
                        continue
                if word_info['boundingBox'][0] > word_info['boundingBox'][4]:
                    continue
                word_infos.append(word_info)
    return word_infos

def check_intersection(box1, box2):
    if (box1[2] - box1[0]) + ((box2[2] - box2[0])) > max(box2[2], box1[2]) - min(box2[0], box1[0]) \
            and (box1[3] - box1[1]) + ((box2[3] - box2[1])) > max(box2[3], box1[3]) - min(box2[1], box1[1]):
        Xc1 = max(box1[0], box2[0])
        Yc1 = max(box1[1], box2[1])
        Xc2 = min(box1[2], box2[2])
        Yc2 = min(box1[3], box2[3])
        intersection_area = (Xc2-Xc1)*(Yc2-Yc1)
        return intersection_area/((box2[3]-box2[1])*(box2[2]-box2[0]))
    else:
        return 0

def try_math(image_path, cls_info):
    title_list = [1, 2, 3]
    title2string = {}
    max_value = 1
    min_value = 0
    max_y = 0
    min_y = 1
    word_infos = ocr_result(image_path)
    for id in title_list:
        if id in cls_info.keys():
            predicted_box = cls_info[id]
            words = []
            for word_info in word_infos:
                word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
                if check_intersection(predicted_box, word_bbox) > 0.5:
                    words.append([word_info["text"], word_bbox[0], word_bbox[1]])
            words.sort(key=lambda x: x[1]+10*x[2])
            word_string = ""
            for word in words:
                word_string = word_string + word[0] + ' '
            title2string[id] = word_string
    if 5 in cls_info.keys():
        plot_area = cls_info[5]
        y_max = plot_area[1]
        y_min = plot_area[3]
        x_board = plot_area[0]
        dis_max = 10000000000000000
        dis_min = 10000000000000000
        for word_info in word_infos:
            word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
            word_text = word_info["text"]
            word_text = re.sub('[^-+0123456789.]', '',  word_text)
            word_text_num = re.sub('[^0123456789]', '', word_text)
            word_text_pure = re.sub('[^0123456789.]', '', word_text)
            if len(word_text_num) > 0 and word_bbox[2] <= x_board+4:
                dis2max = math.sqrt(math.pow((word_bbox[0]+word_bbox[2])/2-x_board, 2)+math.pow((word_bbox[1]+word_bbox[3])/2-y_max, 2))
                dis2min = math.sqrt(math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2) + math.pow(
                    (word_bbox[1] + word_bbox[3]) / 2 - y_min, 2))
                y_mid = (word_bbox[1]+word_bbox[3])/2
                if dis2max <= dis_max:
                    dis_max = dis2max
                    max_y = y_mid
                    max_value = float(word_text_pure)
                    if word_text[0] == '-':
                        max_value = -max_value
                if dis2min <= dis_min:
                    dis_min = dis2min
                    min_y = y_mid
                    min_value = float(word_text_pure)
                    if word_text[0] == '-':
                        min_value = -min_value
        delta_min_max = max_value-min_value
        delta_mark = min_y - max_y
        delta_plot_y = y_min - y_max
        delta = delta_min_max/delta_mark
        if abs(min_y-y_min)/delta_plot_y > 0.05:
            print(abs(min_y-y_min)/delta_plot_y)
            print("Predict the lower bar")
            min_value = min_value + (min_y-y_min)*delta
        if abs(max_y-y_max)/delta_plot_y > 0.05:
            print(abs(max_y-y_max)/delta_plot_y)
            print("Predict the higher bar")
            max_value = max_value + (max_y-y_max)*delta

    return title2string, round(min_value, 2), round(max_value, 2)


def test(image_path, data_type=0, debug=False, suffix=None, min_value_official=None, max_value_official=None):
    image_cls = Image.open(image_path)
    image = cv2.imread(image_path)
    image_draw = Image.fromarray(image)
    with torch.no_grad():
        results = methods['Cls'][2](image, methods['Cls'][0], methods['Cls'][1], debug=False)
        info = results[0]
        tls = results[1]
        brs = results[2]
        plot_area = []
        image_painted, cls_info = GroupCls(image_draw, tls, brs)
        title2string, min_value, max_value = try_math(image_path, cls_info)
        if 5 in cls_info.keys():
            plot_area = cls_info[5][0:4]
        else:
            plot_area = [0, 0, image.shape[1], image.shape[0]]
        if data_type == 0:
            print("Predicted as BarChart")
            results = methods['Bar'][2](image, methods['Bar'][0], methods['Bar'][1], debug=False)
            tls = results[0]
            brs = results[1]
            bar_data = GroupBarRaw(image_draw, tls, brs)
            return bar_data, plot_area, min_value, max_value
        if data_type == 2:
            print("Predicted as PieChart")
            results = methods['Pie'][2](image, methods['Pie'][0], methods['Pie'][1], debug=False)
            cens = results[0]
            keys = results[1]
            image_draw, pie_data = GroupPie(image_draw, cens, keys)
            return pie_data, plot_area, min_value, max_value

        if data_type== 1:
            print("Predicted as LineChart")
            results = methods['Line'][2](image, methods['Line'][0], methods['Line'][1], debug=False, cuda_id=1)
            keys = results[0]
            hybrids = results[1]
            image_painted, quiry, keys, hybrids = GroupQuiryRaw(image_draw, keys, hybrids)
            results = methods['LineCls'][2](image, methods['LineCls'][0], quiry, methods['LineCls'][1], debug=False, cuda_id=1)
            line_data = GroupLineRaw(image_painted, keys, hybrids, results)
            return line_data, plot_area, min_value, max_value


if __name__ == "__main__":
    methods = Pre_load_nets(1, 0)
    tar_path = '/home/t-juluo/evalset_fqa/line/bitmap'
    save_path = '/home/t-juluo/evalset_fqa/line/result.json'
    rs_dict = {}
    images = os.listdir(tar_path)
    for image in tqdm(images):
        path = os.path.join(tar_path, image)
        try:
            data = test(path, 1)
            rs_dict[image] = data
        except:
            continue
    with open(save_path, "w") as f:
        json.dump(rs_dict, f)