import pprint
import os
import json
import argparse
from config import system_configs
from db.datasets import datasets
def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", dest="cfg_file", help="config file", default="CornerNetSimple", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_dir", dest="data_dir", default="./data", type=str)

    args = parser.parse_args()
    return args
args = parse_args()

if args.suffix is None:
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
else:
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
print("cfg_file: {}".format(cfg_file))

with open(cfg_file, "r") as f:
    configs = json.load(f)
configs["system"]["snapshot_name"] = args.cfg_file
system_configs.update_config(configs["system"])

train_split = system_configs.train_split
val_split   = system_configs.val_split
test_split  = system_configs.test_split

split = {
    "training": train_split,
    "validation": val_split,
    "testing": test_split
}[args.split]

print("loading all datasets...")
dataset = system_configs.dataset
print("split: {}".format(split))
db = datasets[dataset](configs["db"], split)

print("system config...")
pprint.pprint(system_configs.full)

print("db config...")
with open("points.json", "r") as f:
    result_json = json.load(f)
pprint.pprint(db.configs)
db_inds = db.db_inds
image_ids = [db.image_ids(ind) for ind in db_inds]
for cls_type in range(1,6):
    db.evaluate(result_json, [cls_type], image_ids)