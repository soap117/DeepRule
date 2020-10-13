import json
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
type_dict = {0:(0,255,0),1:(255,0,0),2:(230,230,0),3:(230,0,233),4:(255,0,255)}
threshold_tag = 0.05

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
    setFont = ImageFont.truetype('C:/windows/fonts/Dengl.ttf', 12)
    fillColor = "#66ccff"

    draw = ImageDraw.Draw(im)
    xy_list = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    xy_list2 = [(x, y), (x, y+h)]
    draw.line(xy_list, fill = type, width = 2)
    draw.line(xy_list2, fill= type , width= 2)
    draw.text((int(x+w/2), int(y+h/2)), u'%.4f' %tag , font=setFont, fill=fillColor)
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

def GroupLine(image, keys_raw, hybrids_raw, plot_area):
    #print(id2name[id])
#    if id2name[id] != 'fc9d4499aa0ce0abc4f31ea63c995511_d3d3LmV4Y2VsbGluZy5pdAk2Mi4xNDkuMTQwLjI4.xls-1-2.png':
#        continue
    keys = []
    for temp in keys_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = int(point[1])
            score = float(point[0])

            keys.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    hybrids = []
    for temp in hybrids_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = int(point[1])
            score = float(point[0])
            hybrids.append({'bbox': bbox, 'category_id': category_id, 'score': score, 'tag': tag})
    for key in keys:
        if key['score'] > 0.35:
            drawLine(image, key['bbox'][0], key['bbox'][1], 3, 3, (int(255*key['score']), 0, 0), key['tag'])
    for key in hybrids:
        if key['score'] > 0.35:
            drawLine(image, key['bbox'][0], key['bbox'][1], 7, 7, (0, int(255*key['score']), 0), key['tag'])
    keys = get_point(keys, 0.35)
    hybrids = get_point(hybrids, 0.35)
    keys = check_cross(keys, hybrids)
    union_result = group_points(keys)
    #image.save(tar_dir + id2name[id])
    data_points = []
    for line in union_result:
        data_line = []
        if len(line) > 1:
            draw_group(line, image)
            for point in line:
                data_line.append([point['bbox'][0], point['bbox'][1]])
            data_line = get_data(data_line, plot_area)
            data_points.append(data_line)
    pre_data[id2name[id]] = data_points
    #image.save(tar_dir + 'group_' +id2name[id])
with open('results/data_rs.json', "w") as f:
    json.dump(pre_data, f)