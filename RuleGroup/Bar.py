import os
import json
import numpy
import math
from PIL import Image, ImageDraw, ImageFont
import copy
from tqdm import tqdm
type_dict = {0:(0,255,0),1:(255,0,0),2:(230,230,0),3:(230,0,233),4:(255,0,255),5:(125, 255, 233)}
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

def drawData(im, x, y, w, h, datum):
    draw = ImageDraw.Draw(im)
    fillColor = "#66ccff"
    draw.text((int(x), int(y-8)), u'%.2f' % datum, fill=fillColor)
    del draw

def draw_group(groups, im):
    for group in groups:
        drawLine(im, group[0], group[1], group[2]-group[0], group[3]-group[1], (0, 0, 255))

def draw_data(groups, im, min_value, max_value, plot_area):
    for group in groups:
        x = group[0]
        frac_x = (x - plot_area[0]) / (plot_area[2] - plot_area[0])
        y = group[3] - group[1]
        frac_y = (y) / (plot_area[3] - plot_area[1])
        drawData(im, group[0], group[1], group[2]-group[0], group[3]-group[1], (max_value - min_value) * frac_y + min_value)

def get_data(groups, plot_area):
    data = []
    for group in groups:
        x = group[0]
        frac_x = (x - plot_area[0]) / (plot_area[2] - plot_area[0])
        y = group[3] - group[1]
        frac_y = (y) / (plot_area[3] - plot_area[1])
        data.append([frac_x, frac_y])

    data.sort(key = lambda x: x[0]*1000+x[1])
    data_pure = []
    for datum in data:
        data_pure.append(datum[1])
    return data_pure

def get_data_divided(groups, plot_area):
    data_divided = []
    for gset in groups:
        data = []
        for group in gset:
            x = group[0]
            frac_x = (x - plot_area[0]) / (plot_area[2] - plot_area[0])
            y = group[3] - group[1]
            frac_y = (y) / (plot_area[3] - plot_area[1])
            data.append([frac_x, frac_y])

        data.sort(key = lambda x: x[0]*1000+x[1])
        data_pure = []
        for datum in data:
            data_pure.append(datum[1])
        data_divided.append(data_pure)
    return data_divided

def scale_adjust(data, x_min, x_max, y_min, y_max):
    true_data = []
    for point in data:
        true_x = (x_max - x_min) * point[0] + x_min
        true_y = (y_max - y_min) * point[0] + y_min
        true_data.append([true_x, true_y])
    return true_data


def cal_dis(a, b):
    return -(a['bbox'][0]-b['bbox'][0]+0.1*(a['bbox'][1]-b['bbox'][1]))


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
            if br_key['bbox'][0] > tl_key['bbox'][0] + 4 and br_key['bbox'][1] > tl_key['bbox'][1] + 4:
                dis = cal_dis(tl_key, br_key)
                score = br_key['score']
                #dis_score = dis * math.pow(1 - score, 1/16)
                dis_score = dis
                if dis_score < min_dis_score:
                    min_dis_score = dis_score
                    target_br = br_key
        if target_br != None:
            pairs.append([tl_key['bbox'][0], tl_key['bbox'][1], target_br['bbox'][0], target_br['bbox'][1]])
    return pairs

class UnionFindSet(object):
    def __init__(self, data_list):
        self.father_dict = {}
        self.size_dict = {}
        for i in range(len(data_list)):
            self.father_dict[i] = i
            self.size_dict[i] = 1

    def find_head(self, ID):

        father = self.father_dict[ID]
        if(ID != father):
            father = self.find_head(father)
        self.father_dict[ID] = father
        return father

    def is_same_set(self, ID_a, ID_b):
        return self.find_head(ID_a) == self.find_head(ID_b)

    def union(self, ID_a, ID_b):
        if ID_a is None or ID_a is None:
            return

        a_head = self.find_head(ID_a)
        b_head = self.find_head(ID_b)

        if(a_head != b_head):
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if(a_set_size >= b_set_size):
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size

def get_color_dis(bbox_a, bbox_b, image):
    area_a = image[int(bbox_a[1]+1):int(bbox_a[3]-1), int(bbox_a[0]+1):int(bbox_a[2]-1)].mean(axis=0).mean(axis=0)
    area_b = image[int(bbox_b[1]+1):int(bbox_b[3]-1), int(bbox_b[0]+1):int(bbox_b[2]-1)].mean(axis=0).mean(axis=0)
    mean_dis = numpy.abs(area_a-area_b).mean()/255
    return mean_dis

def divided_by_color(groups, raw_image):
    raw_image.save('debug.png')
    threshold_color = 0.1
    raw_image = numpy.array(raw_image)
    dis_list = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            color_dis = get_color_dis(groups[i], groups[j], raw_image)
            dis_list.append([color_dis, i, j])
    dis_list.sort(key=lambda x:x[0])
    unionset = UnionFindSet(groups)
    for dis_pair in dis_list:
        if dis_pair[0] > threshold_color:
            break
        unionset.union(dis_pair[1], dis_pair[2])
    grouped = {}
    for i in range(len(groups)):
        if unionset.size_dict[i] > 0:
            grouped[i] = []
    for i in range(len(groups)):
        grouped[unionset.father_dict[i]].append(groups[i])
    grouped = [x for x in grouped.values() if len(x)>0]
    return grouped

def GroupBar(image, tls_raw, brs_raw, plot_area, min_value, max_value):
    image_raw = copy.deepcopy(image)
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
    tls = get_point(tls, 0.4)
    brs = get_point(brs, 0.4)
    for key in tls:
        drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (int(255 * key['score']), 0, 0))
    for key in brs:
        drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (0, int(255 * key['score']), 0))
    #image.save(tar_dir + id2name[id])
    info = {}
    if len(tls) > 0:
        for tar_id in range(1):
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
            draw_group(groups, image)
            data = get_data(groups, plot_area)
            draw_data(groups, image, min_value, max_value, plot_area)
            groups_divided = divided_by_color(groups, image_raw)
            data_divided = get_data_divided(groups_divided, plot_area)
    return image, data_divided

def GroupBarRaw(image, tls_raw, brs_raw):
    image_raw = copy.deepcopy(image)
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
    tls = get_point(tls, 0.4)
    brs = get_point(brs, 0.4)
    for key in tls:
        drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (int(255 * key['score']), 0, 0))
    for key in brs:
        drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (0, int(255 * key['score']), 0))
    #image.save(tar_dir + id2name[id])
    info = {}
    if len(tls) > 0:
        for tar_id in range(1):
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
            draw_group(groups, image)
    return groups