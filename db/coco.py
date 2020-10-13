import sys

import os
import json
import numpy as np
import pickle
import copy
from tqdm import tqdm
from db.detection import DETECTION
from config import system_configs
from pycocotool.coco import COCO
from pycocotool.cocoeval import COCOeval


class MSCOCO(DETECTION):
    def __init__(self, db_config, split):
        super(MSCOCO, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainval": "trainval2014",
            "minival": "minival2014",
            "testdev": "testdev2017",
            "trainval(FLIP)": "trainval2014(FLIP)",
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "coco")
        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instances_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "coco"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "coco_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)

                    categories.append(category)

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[2], info[3], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]


class Pie(DETECTION):
    def __init__(self, db_config, split):
        super(Pie, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "train2019",
            "valchart": "val2019",
            "testchart": "test2019"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "pie")

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instancesPie(1008)_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "pie"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "pie_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bboxes.append(bbox)
                    categories.append(category)

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[2], info[3], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]


class Line(DETECTION):
    def __init__(self, db_config, split):
        super(Line, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "train2019",
            "valchart": "val2019",
            "testchart": "test2019"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "line")

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instancesLine(1023)_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "line"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "line_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                max_len = 0
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bboxes.append(bbox)
                    categories.append(category)
                    max_len = max(max_len, len(bbox))
            for ind_bbox in range(len(bboxes)):
                if len(bboxes[ind_bbox]) < max_len:
                    bboxes[ind_bbox] = np.pad(bboxes[ind_bbox], (0, max_len - len(bboxes[ind_bbox])), 'constant',
                                              constant_values=(0, 0))
            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if len(bboxes) == 0 or len(categories) == 0:
                self._detections[image_id] = None
            else:
                self._detections[image_id] = (bboxes, categories)

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return copy.deepcopy(detections)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]
                    tag = info[1]
                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score)),
                        "tag" : float(tag)
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]


class LineCls(DETECTION):
    def __init__(self, db_config, split):
        super(LineCls, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "train2019",
            "valchart": "val2019",
            "testchart": "test2019"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "line")

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instancesLineCls(1119)_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "line_cls"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "line_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            categories = []
            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                max_len = 0
            if len(annotations) == 0:
                self._detections[image_id] = (None, None)
            else:
                bbox = np.array(annotations[0]["bbox"])
                self._detections[image_id] = (bbox[0], bbox[1])

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return copy.deepcopy(detections)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]
                    tag = info[1]
                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score)),
                        "tag" : float(tag)
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]

class LineClsReal(DETECTION):
    def __init__(self, db_config, split):
        super(LineClsReal, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "train2019",
            "valchart": "val2019",
            "testchart": "test2019"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "line")

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instancesLineClsReal(1119)_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "line_cls"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "line_real_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            categories = []
            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                max_len = 0
            if len(annotations) == 0:
                self._detections[image_id] = None
            else:
                bbox = np.array(annotations[0]["bbox"])
                self._detections[image_id] = bbox

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return copy.deepcopy(detections)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]
                    tag = info[1]
                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score)),
                        "tag" : float(tag)
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]

class Bar(DETECTION):
    def __init__(self, db_config, split):
        super(Bar, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "train2019",
            "valchart": "val2019",
            "testchart": "test2019"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "bar")

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instancesBar(1031)_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "bar"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "chart_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)

                    categories.append(category)

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[2], info[3], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]


class Cls(DETECTION):
    def __init__(self, db_config, split):
        super(Cls, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir
        tar_data_type = system_configs.tar_data_dir
        self._split = split
        self._dataset = {
            "trainchart": "train2019",
            "valchart": "val2019",
            "testchart": "test2019"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, tar_data_type)

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instances%s(1031)_{}.json" %tar_data_type.capitalize())
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = tar_data_type
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "cls_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if system_configs.tar_data_dir == 'cls':
            if not os.path.exists(self._cache_file):
                print("No cache file found...")
                self._extract_data()
                with open(self._cache_file, "wb") as f:
                    pickle.dump([self._detections, self._image_ids, self._clss, self._offsets], f)
            else:
                with open(self._cache_file, "rb") as f:
                    self._detections, self._image_ids, self._clss, self._offsets = pickle.load(f)
        else:
            self._extract_data()

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        try:
            self._clss = [
                self._coco.loadImgs(img_id)[0]["data_type"]
                for img_id in coco_image_ids
            ]
            self._offsets = [
                self._coco.loadImgs(img_id)[0]["offset"]
                for img_id in coco_image_ids
            ]
        except:
            print("Without Groud Truth")
        self._detections = {}
        if system_configs.tar_data_dir == 'cls':
            for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
                image = self._coco.loadImgs(coco_image_id)[0]
                bboxes = []
                categories = []

                for cat_id in self._cat_ids:
                    annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                    annotations = self._coco.loadAnns(annotation_ids)
                    category = self._coco_to_class_map[cat_id]
                    for annotation in annotations:
                        bbox = np.array(annotation["bbox"])
                        bbox[[2, 3]] += bbox[[0, 1]]
                        bboxes.append(bbox)

                        categories.append(category)

                bboxes = np.array(bboxes, dtype=float)
                categories = np.array(categories, dtype=float)
                if bboxes.size == 0 or categories.size == 0:
                    self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
                else:
                    self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def image_info(self, ind):
        image_id = self._image_ids[ind]
        cls = self._clss[ind]
        offset = self._offsets[ind]
        return cls, offset

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[2], info[3], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]


class SKU(DETECTION):
    def __init__(self, db_config, split):
        super(SKU, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainsku": "trainsku",
            "valsku": "valsku",
            "testsku": "testsku"
        }[self._split]
        self._coco_dir = os.path.join(data_dir, "SKU110K")
        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instances_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images")
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "sku"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "sku_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)

                    categories.append(category)

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[2], info[3], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]


class Chart(DETECTION):
    def __init__(self, db_config, split):
        super(Chart, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "train2019",
            "valchart": "val2019",
            "testchart": "test2019"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "chart")

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instances_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "chart"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "chart_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)

                    categories.append(category)

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points_pure(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[2], info[3], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]


class ChartXY(DETECTION):
    def __init__(self, db_config, split):
        super(ChartXY, self).__init__(db_config)
        data_dir = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "trainchart",
            "valchart": "valchart",
            "testchart": "testchart"
        }[self._split]

        self._coco_dir = os.path.join(data_dir, "chart")

        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instances_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "chart"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "chart_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                category = self._coco_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)

                    categories.append(category)

            bboxes = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def convert_to_coco_points(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for info in all_bboxes[image_id][cls_ind]:
                    bbox = [info[3], info[4], 6, 6]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    score = info[0]

                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]