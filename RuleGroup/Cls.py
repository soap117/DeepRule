import os
import json
import numpy
import math
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
type_dict = {0:(0,255,0),1:(255,0,0),2:(230,230,0),3:(130,0,233),4:(255,0,255),5:(125, 255, 233)}
def get_point(points, threshold):
    count = 0
    points_clean = []
    for point in points:
        if point['score'] > threshold:
            count += 1
            points_clean.append(point)
    return points_clean


def drawLine(im, x, y, w, h, type):
    '''
    在图片上绘制矩形图
    :param im: 图片
    :param width: 矩形宽占比
    :param height: 矩形高占比
    :return:
    '''

    draw = ImageDraw.Draw(im)
    xy_list = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    xy_list2 = [(x, y), (x, y+h)]
    draw.line(xy_list, fill = type, width = 2)
    draw.line(xy_list2, fill= type , width= 2)
    del draw

def drawRange(group, im):
    setFont = ImageFont.truetype('C:/windows/fonts/Dengl.ttf', 15)
    fillColor = "#ee0000"

    draw = ImageDraw.Draw(im)
    draw.text((10, 10), u'X range %.4f : %.4f' % (group[0], group[2]) , font=setFont, fill=fillColor)
    draw.text((10, 30), u'Y range %.4f : %.4f' % (group[1], group[3]), font=setFont, fill=fillColor)
    del draw

def drawInfo(type, im):
    setFont = ImageFont.truetype('C:/windows/fonts/Dengl.ttf', 15)
    fillColor = "#ee0000"
    draw = ImageDraw.Draw(im)
    draw.text((10, 60), u'%s' % (type) , font=setFont, fill=fillColor)
    del draw


def draw_group(groups, im, color):
    for group in groups:
        drawLine(im, group[0], group[1], group[2]-group[0], group[3]-group[1], color)


def cal_dis(a, b):
    return -(a['bbox'][0]-b['bbox'][0]+0.1*(a['bbox'][1]-b['bbox'][1]))

def CountIoU(box1, box2):
    if (box1[2] - box1[0]) + ((box2[2] - box2[0])) > max(box2[2], box1[2]) - min(box2[0], box1[0]) \
            and (box1[3] - box1[1]) + ((box2[3] - box2[1])) > max(box2[3], box1[3]) - min(box2[1], box1[1]):
        Xc1 = max(box1[0], box2[0])
        Yc1 = max(box1[1], box2[1])
        Xc2 = min(box1[2], box2[2])
        Yc2 = min(box1[3], box2[3])
        intersection_area = (Xc2-Xc1)*(Yc2-Yc1)
        return intersection_area, intersection_area/((box2[3]-box2[1])*(box2[2]-box2[0]))
    else:
        return -1, 0

def estimate_zero_line(br_keys):
    ys_sum = 0
    score_sum = 0
    for key in br_keys:
        ys_sum += key['score']*key['bbox'][1]
        score_sum += key['score']
    mean = ys_sum/score_sum
    temp = 0
    for key in br_keys:
        temp += math.pow(key['score']-mean, 2)*key['score']
    temp /= score_sum
    new_ys = []
    std = math.sqrt(temp)
    for y in br_keys:
        if abs(y['bbox'][1]-mean) < std:
            new_ys.append(y['bbox'][1])
    return numpy.array(new_ys).mean()


def group_point(tl_keys, br_keys):
    pairs = []
    for tl_key in tl_keys:
        min_dis_score = 9999999999
        target_br = None
        for br_key in br_keys:
            if br_key['category_id'] == tl_key['category_id'] and br_key['bbox'][0] > tl_key['bbox'][0] + 3 and br_key['bbox'][1] > tl_key['bbox'][1] + 3:
                dis = cal_dis(tl_key, br_key)
                score = br_key['score']
                dis_score = dis * math.pow(1 - score, 1/4)
                if dis_score < min_dis_score:
                    min_dis_score = dis_score
                    target_br = br_key
        if target_br != None:
            pairs.append([tl_key['bbox'][0], tl_key['bbox'][1], target_br['bbox'][0], target_br['bbox'][1], math.sqrt(tl_key['score']*target_br['score'])])
    return pairs

def GroupCls(image, tls_raw, brs_raw):
    tls = []
    for temp in tls_raw.values():
        for point in temp:
            bbox = [point[2], point[3], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[1])
            score = float(point[0])
            tls.append({'bbox':bbox, 'category_id': category_id, 'score': score})
    brs = []
    for temp in brs_raw.values():
        for point in temp:
            bbox = [point[2], point[3], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[1])
            score = float(point[0])
            brs.append({'bbox': bbox, 'category_id': category_id, 'score': score})
    tls = get_point(tls, 0.3)
    brs = get_point(brs, 0.3)
    for key in tls:
        drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (int(255 * key['score']), 0, 0))
    for key in brs:
        drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (0, int(255 * key['score']), 0))
    #image.save(tar_dir + id2name[id])
    info = {}
    if len(tls) > 0:
        for tar_id in range(6):
            tl_same = []
            br_same = []
            for tl in tls:
                if tl['category_id'] == tar_id:
                    tl_same.append(tl)
            for br in brs:
                if br['category_id'] == tar_id:
                    br_same.append(br)
            #zero_y = estimate_zero_line(brs)
            groups = group_point(tl_same, br_same)
            max_score = 0
            tar_group = None
            for group in groups:
                if group[4] > max_score:
                    max_score = group[4]
                    tar_group = group
            if tar_group == None:
                continue
            draw_group([tar_group], image, type_dict[tar_id])
            info[tar_id] = tar_group
    return image, info
