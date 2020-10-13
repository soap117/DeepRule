import json
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
type_dict = {0:(0,255,0),1:(255,0,0),2:(230,230,0),3:(230,0,233),4:(255,0,255)}
threshold_tag = 0.085

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


def drawLine(im, x, y, w, h, type, tag):
    '''
    在图片上绘制矩形图
    :param im: 图片
    :param width: 矩形宽占比
    :param height: 矩形高占比
    :return:
    '''
    fillColor = "#66ccff"

    draw = ImageDraw.Draw(im)
    xy_list = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    xy_list2 = [(x, y), (x, y+h)]
    draw.line(xy_list, fill = type, width = 2)
    draw.line(xy_list2, fill= type , width= 2)
    draw.text((int(x+w/2), int(y+h/2)), u'%.4f' %tag , fill=fillColor)
    del draw

def get_point(points, threshold):
    count = 0
    points_clean = []
    for point in points:
        if point['score'] > threshold:
            count += 1
            points_clean.append(point)
    return points_clean


def compute_tag_dis(key1, key2):
    return abs(key1['tag']- key2['tag'])


def get_key(a):
    return a[1]


def group_points(keys):
    dis_array = {}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            dis_array[(i, j)] = compute_tag_dis(keys[i], keys[j])
    dis_array = list(dis_array.items())
    dis_array.sort(key=get_key)
    unionset = UnionFindSet(keys)
    for pair in dis_array:
        if pair[1] > threshold_tag:
            break
        if not keys[pair[0][0]]['is_cross'] and not keys[pair[0][1]]['is_cross']:
            unionset.union(pair[0][0], pair[0][1])
    group = {}
    for i in range(len(keys)):
        if unionset.size_dict[i] > 0:
            group[i] = []
    for i in range(len(keys)):
        group[unionset.father_dict[i]].append(i)
    #print(group)
    group = list(group.values())
    for line in  group:
        for i in range(len(line)):
            line[i] = keys[line[i]]
    return group

def get_data(line, plot_area):
    data = []
    for group in line:
        x = group[0]
        frac_x = (x - plot_area[0]) / (plot_area[2] - plot_area[0])
        y = plot_area[3] - group[1]
        frac_y = (y) / (plot_area[3] - plot_area[1])
        data.append([frac_x, frac_y])

    data.sort(key = lambda x: x[0]*100+x[1])
    return data

def draw_group(line, im):
    line.sort(key=lambda x: x['bbox'][0])
    draw = ImageDraw.Draw(im)
    xy_list = []
    for key in line:
        xy_list.append((key['bbox'][0], key['bbox'][1]))
    draw.line(xy_list, fill=(0, 255, 0), width=2)
    del draw

def check_cross(keys, hybrids):
    for key in keys:
        key['is_cross'] = False
    for hybrid in hybrids:
        border = [hybrid['bbox'][0]-4, hybrid['bbox'][1]-4, hybrid['bbox'][0]+4, hybrid['bbox'][1]+4]
        for key in keys:
            if key['bbox'][0] >= border[0] and key['bbox'][1] >= border[1] and key['bbox'][0] <= border[2] and key['bbox'][1] <= border[3]:
                key['is_cross'] = True
    return keys

def quiry_for_hybrid(keys):
    keys.sort(key = lambda x:x['bbox'][0])
    quirys = []
    for ind, key in enumerate(keys):
        if key['is_cross']:
            rp_ind_s = ind+1
            for rp_ind_s in range(ind+1, len(keys)):
                if abs(keys[rp_ind_s]['bbox'][0]-key['bbox'][0])>4:
                    break
            rp_ind_e = min(rp_ind_s + 1, len(keys))
            for rp_ind_e in range(rp_ind_s, len(keys)):
                if abs(keys[rp_ind_e]['bbox'][0]-keys[rp_ind_s]['bbox'][0])>4:
                    break
            for r_ind in range(rp_ind_s, rp_ind_e):
                quiry_pair = [[], []]
                quiry_pair[0] = [key['bbox'][0], key['bbox'][1]]
                quiry_pair[1] = [keys[r_ind]['bbox'][0], keys[r_ind]['bbox'][1]]
                quirys.append(quiry_pair)
            lp_ind_s = ind - 1
            for lp_ind_s in range(ind - 1, -1, -1):
                if abs(keys[lp_ind_s]['bbox'][0] - key['bbox'][0])>4:
                    break
            lp_ind_e = max(lp_ind_s - 1, -1)
            for lp_ind_e in range(lp_ind_s, -1, -1):
                if abs(keys[lp_ind_e]['bbox'][0] - keys[lp_ind_s]['bbox'][0])>4:
                    break
            for l_ind in range(lp_ind_s, lp_ind_e, -1):
                quiry_pair = [[], []]
                quiry_pair[1] = [key['bbox'][0], key['bbox'][1]]
                quiry_pair[0] = [keys[l_ind]['bbox'][0], keys[l_ind]['bbox'][1]]
                quirys.append(quiry_pair)
    return quirys

def drawData(im, x, y, w, h, datum):
    draw = ImageDraw.Draw(im)
    fillColor = "#00eeff"
    draw.text((int(x), int(y-8)), u'%.2f' % datum, fill=fillColor)
    del draw


def GroupQuiry(image, keys_raw, hybrids_raw, plot_area, min_value, max_value):
    keys = []
    for temp in keys_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])

            keys.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    hybrids = []
    for temp in hybrids_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])
            hybrids.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    for key in keys:
        if key['score'] > 0.4:
            drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (int(255 * key['score']), 0, 0), key['tag'])
            y = plot_area[3] - key['bbox'][1]
            frac_y = (y) / (plot_area[3] - plot_area[1])
            drawData(image, key['bbox'][0], key['bbox'][1], 3, 3, frac_y*(max_value-min_value)+min_value)
    for key in hybrids:
        if key['score'] > 0.4:
            drawLine(image, key['bbox'][0], key['bbox'][1], 7, 7, (0, int(255 * key['score']), 0), key['tag'])
    keys = get_point(keys, 0.4)
    hybrids = get_point(hybrids, 0.4)
    keys = check_cross(keys, hybrids)
    quiries = quiry_for_hybrid(keys)
    return image, quiries, keys, hybrids

def GroupQuiryRaw(image, keys_raw, hybrids_raw):
    keys = []
    for temp in keys_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])

            keys.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    hybrids = []
    for temp in hybrids_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])
            hybrids.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    keys = get_point(keys, 0.4)
    hybrids = get_point(hybrids, 0.4)
    keys = check_cross(keys, hybrids)
    quiries = quiry_for_hybrid(keys)
    return image, quiries, keys, hybrids