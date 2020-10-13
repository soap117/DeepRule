import json
import argparse
import matplotlib
matplotlib.use("Agg")
from config import system_configs
from db.datasets import datasets
def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    if args.suffix is None:
        cfg_file = "/mnt/d/CornerNet-CloudVersion/config/CornerNet.json"
    else:
        cfg_file = "/mnt/d/CornerNet-CloudVersion/config/CornerNet.json"
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }[args.split]

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))
    db = datasets[dataset](configs["db"], split)
    categories = db.configs["categories"]
    if db.split != "trainval":
        db_inds = db.db_inds
    else:
        db_inds = db.db_inds[:5000]
    image_ids = [db.image_ids(ind) for ind in db_inds]
    result_json = json.load(open("./test/results.json", "r"))
    for i in range(5):
        cls_ids = list([i+1])
        db.evaluate(result_json, cls_ids, image_ids)