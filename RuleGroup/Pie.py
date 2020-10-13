import json
import os
import math
from functools import cmp_to_key
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
type_dict = {0:(0,255,0),1:(255,0,0),2:(230,230,0),3:(230,0,233),4:(255,0,255)}
def drawArc(im, group, type):
    '''
    在图片上绘制矩形图
    :param im: 图片
    :param width: 矩形宽占比
    :param height: 矩形高占比
    :return:
    '''
    draw = ImageDraw.Draw(im)
    xy_list1 = [group[0], group[1]]
    xy_list2 = [group[0], group[2]]
    draw.line(xy_list1, fill=type_dict[type], width=2)
    draw.line(xy_list2, fill=type_dict[type], width=2)
    theta1 = math.degrees(math.acos((group[1][0]-group[0][0])/(math.sqrt(math.pow(group[1][0]-group[0][0], 2) + math.pow(group[1][1]-group[0][1], 2)))))
    if group[1][1]-group[0][1] > 0:
        theta1 = 360-theta1
    theta2 = math.degrees(math.acos((group[2][0] - group[0][0]) / (
        math.sqrt(math.pow(group[2][0] - group[0][0], 2) + math.pow(group[2][1] - group[0][1], 2)))))
    if group[2][1] - group[0][1] > 0:
        theta2 = 360-theta2
    r = math.sqrt(math.pow(group[2][0] - group[0][0], 2) + math.pow(group[2][1] - group[0][1], 2))
    xy = (group[0][0]-r, group[0][1]-r, group[0][0]+r, group[0][1]+r)
    draw.arc(xy, 360-theta2, 360-theta1, fill=type_dict[type], width=3)
    del draw

def get_point(points, threshold):
    count = 0
    points_clean = []
    for point in points:
        if point['score'] > threshold:
            count += 1
            points_clean.append(point)
    return points_clean


def cal_dis(a, b):
    return math.sqrt(math.pow(a['bbox'][0]-b['bbox'][0], 2)+math.pow(a['bbox'][1]-b['bbox'][1], 2))


def cross(a):
    center_x = center['bbox'][0]
    center_y = center['bbox'][1]
    left_x = a['bbox'][0]
    left_y = a['bbox'][1]
    x1 = left_x - center_x
    y1 = left_y - center_y
    theta_y = math.degrees(math.acos((-y1 / math.sqrt(x1 * x1 + y1 * y1))))
    if x1 < 0:
        theta_y = 360 - theta_y
    return theta_y

def pair_one(center_point, key_points):
    global center
    center = center_point
    key_points = sorted(key_points, key=cross, reverse=True)
    groups = []
    for i in range(len(key_points)):
        score = (center_point['score'] + key_points[i]['score'] + key_points[(i+1)%len(key_points)]['score'])/3
        groups.append([tuple(center_point['bbox'][0:2]), tuple(key_points[i]['bbox'][0:2]), tuple(key_points[(i+1)%len(key_points)]['bbox'][0:2]), score])
    return groups


def pair_multi(center_points, key_points, r, threshold):
    global center
    center = center_points[0]
    key_points = sorted(key_points, key=cross, reverse=True)
    groups = []
    for i in range(len(key_points)):
        key_point = key_points[i]
        for j in range(len(center_points)):
            r_ = cal_dis(key_point, center_points[j])
            if abs((r_-r)/r) <= threshold:
                tar_center = center_points[j]
                break
        r_ = cal_dis(key_points[(i+1)%len(key_points)], tar_center)
        if abs((r_ - r) / r) <= threshold:
            score = (tar_center['score'] + key_points[i]['score'] + key_points[(i + 1) % len(key_points)][
                'score']) / 3
            groups.append([tuple(tar_center['bbox'][0:2]), tuple(key_points[i]['bbox'][0:2]), tuple(key_points[(i+1)%len(key_points)]['bbox'][0:2]), score])
    return groups


def get_count(all_r, threshold):
    max_count = 0
    for r_base in all_r:
        count = 0
        for r in all_r:
            if abs((r-r_base)/r_base) <= threshold:
                count += 1
        if count > max_count:
            max_count = count
            record_r = r_base
    return record_r, max_count

def binary_search(all_r, tar_count):
    lt = 0.05
    lr = 0.2
    record_r, count = get_count(all_r, (lt+lr)/2)
    while (lr-lt) > 1e-3:
        if count < tar_count:
            lt = (lr+lt)/2
        else:
            lr = (lr+lt)/2
        record_r, count = get_count(all_r, (lt + lr) / 2)
    return record_r, lt

def estimatie_r(centers, keys):
    all_r = []
    for center in centers:
        for key in keys:
            all_r.append(cal_dis(center, key))
    target_count = len(keys)
    r, threshold = binary_search(all_r, target_count)
    return r, threshold


def check_key(k, keys, centers):
    key = keys[k]
    left_key = keys[(k-1)%len(keys)]
    right_key = keys[(k+1)%len(keys)]
    flag = False
    for center in centers:
        r = cal_dis(key, center)
        rl = cal_dis(left_key, center)
        rr = cal_dis(right_key, center)
        if abs((rl-r)/r) < 0.1 or abs((rr-r)/r) < 0.1:
            flag = True
            break
    return flag


def check_center(keys, center):
    flag = False
    for i in range(len(keys)):
        rl = cal_dis(keys[i], center)
        rr = cal_dis(keys[(i+1)%len(keys)], center)
        if abs((rl-rr)/rr) < 0.1:
            flag = True
            break
    return flag



def filter(centers, keys):
    global center
    center = centers[0]
    keys = sorted(keys, key=cross, reverse=True)
    for i in range(len(keys)-1, -1, -1):
        if not check_key(i, keys, centers):
            keys.remove(keys[i])
    for i in range(len(centers)-1, -1, -1):
        if not check_center(keys, centers[i]):
            centers.remove(centers[i])
    return centers, keys


def get_anno(groups, image_id, category_id):
    annos = []
    for group in groups:
        anno = {}
        anno['image_id'] = image_id
        anno['category_id'] = category_id
        segem = []
        xa = group[1][0] - group[0][0]
        ya = group[1][1] - group[0][1]
        xb = group[2][0] - group[0][0]
        yb = group[2][1] - group[0][1]
        cross = xa * yb - xb * ya
        r = math.sqrt(xa * xa + ya * ya)
        if cross != 0:
            xm = (xa + xb) / 2
            ym = (ya + yb) / 2
            ration = r / math.sqrt(xm * xm + ym * ym)
            xm = ration * xm
            ym = ration * ym
            if cross > 0:
                xm = -xm
                ym = -ym
        else:
            xm = ya
            ym = -xa
            if ya < 0:
                xm = -xm
                ym = -ym
        xm_l = (xb + xm) / 2
        ym_l = (yb + ym) / 2
        xm_r = (xa + xm) / 2
        ym_r = (ya + ym) / 2
        ration = r / math.sqrt(xm_l * xm_l + ym_l * ym_l)
        xm_l = ration * xm_l
        ym_l = ration * ym_l
        ration = r / math.sqrt(xm_r * xm_r + ym_r * ym_r)
        xm_r = ration * xm_r
        ym_r = ration * ym_r
        xm_ = round(xm + group[0][0], 2)
        ym_ = round(ym + group[0][1], 2)
        xm_l = round(xm_l + group[0][0], 2)
        ym_l = round(ym_l + group[0][1], 2)
        xm_r = round(xm_r + group[0][0], 2)
        ym_r = round(ym_r + group[0][1], 2)
        segem.append([group[0][0], group[0][1], group[2][0], group[2][1], xm_l, ym_l, xm_, ym_, xm_r, ym_r, group[1][0], group[1][1]])
        anno['bbox'] = [group[0][0]-0.5*r, group[0][1]-0.5*r, group[0][0]+0.5*r, group[0][1]+0.5*r]
        anno['segmentation'] = segem
        anno['score'] = group[3]
        annos.append(anno)
    return annos

def ekey(x):
    return x[0]

def GroupPie(image, tls_raw, brs_raw):
    centers = []
    for temp in tls_raw.values():
        for point in temp:
            bbox = [point[2], point[3], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[1])
            score = float(point[0])
            centers.append({'bbox': bbox, 'category_id': category_id, 'score': score})
    keys = []
    for temp in brs_raw.values():
        for point in temp:
            bbox = [point[2], point[3], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[1])
            score = float(point[0])
            keys.append({'bbox': bbox, 'category_id': category_id, 'score': score})
    centers = get_point(centers, 0.30)
    keys = get_point(keys, 0.30)
    if len(centers) > 0:
        centers, keys = filter(centers, keys)
        if len(centers) == 1:
            groups = pair_one(centers[0], keys)
            for group in groups:
                drawArc(image, group, 0)
        if len(centers) > 1:
            r, threshold = estimatie_r(centers, keys)
            groups = pair_multi(centers, keys, r, threshold)
            for group in groups:
                drawArc(image, group, 0)
        data_rs = []
        for group in groups:
            center_x = group[0][0]
            center_y = group[0][1]
            left_x = group[2][0]
            left_y = group[2][1]
            right_x = group[1][0]
            right_y = group[1][1]
            x1 = left_x - center_x
            y1 = left_y - center_y
            x2 = right_x - center_x
            y2 = right_y - center_y
            theta = math.degrees(math.acos(
                max(min((x1 * x2 + y1 * y2) / math.sqrt(x1 * x1 + y1 * y1) / math.sqrt(x2 * x2 + y2 * y2), 1), -1)))
            cross = x1 * (y2) - x2 * (y1)
            if cross < 0:
                theta = 360 - theta
            theta_y = math.degrees(math.acos((-y1 / math.sqrt(x1 * x1 + y1 * y1))))
            if x1 < 0:
                theta_y = 360 - theta_y
            data_rs.append([theta_y, theta])
        data_rs.sort(key=ekey)
        data_pure = []
        for datum in data_rs:
            data_pure.append(datum[1])
        return image, data_pure

