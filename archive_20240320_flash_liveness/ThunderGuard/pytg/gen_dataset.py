import os
import argparse
import random
from shutil import copyfile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--from_path', type=str, default=r"F:\tg_video\dataset\dnormal", help='数据文件地址')
parser.add_argument('--to_path', type=str, default="../data/sample/delta", help='数据文件地址')
parser.add_argument('--opt', type=str, default='random', help='划分训练测试集的方法')
parser.add_argument('--train_percent', type=float, default=0.9, help='训练测试集的比例')
args = parser.parse_args()


def copy_to(from_file_list, to_path):
    for from_file in tqdm(from_file_list):
        file = os.path.basename(from_file)
        pre_name = str(file2tag.get(file, 50)) + '_'
        for last_name in [".txt", ".jpg", "_d.jpg"]:
            copyfile(from_file + last_name, os.path.join(to_path, pre_name + file + last_name))


def random_split():
    for key, value in file_list_map.items():

        if key == "7":
            # 全部放到训练集
            copy_to(value, train_path)
        else:
            timestamp_list = list(set([get_timestamp(os.path.basename(f)) for f in value]))
            random.shuffle(timestamp_list)
            s = int(len(timestamp_list) * args.train_percent)
            train_timestamp_set = set(timestamp_list[:s])
            train_value = list(filter(lambda a: get_timestamp(os.path.basename(a)) in train_timestamp_set, value))
            copy_to(train_value, train_path)

            test_timestamp_set = set(timestamp_list[s:])
            test_value = list(filter(lambda a: get_timestamp(os.path.basename(a)) in test_timestamp_set, value))
            copy_to(test_value, test_path)


def get_timestamp(file_name):
    tmp = file_name.split('_')
    timestamp = tmp[1] if len(tmp) == 7 else tmp[0]
    return int(timestamp)


if __name__ == '__main__':
    dir_list = os.listdir(args.from_path)
    file_list_map = dict()
    for dir in dir_list:
        file_path = os.path.join(args.from_path, dir)
        if os.path.isdir(file_path):
            sub_file_list = os.listdir(file_path)
            sub_file_list = list(filter(lambda a: a.endswith(".txt"), sub_file_list))
            sub_file_list = list(map(lambda a: os.path.join(file_path, a.replace(".txt", "")), sub_file_list))
            for file in sub_file_list:
                flag = file.split('_')[-1]
                file_list_map.setdefault(flag, []).append(file)

    sample_meta_path = os.path.join(args.from_path, "sample_meta.txt")
    file2tag = {}
    if os.path.exists(sample_meta_path):
        with open(sample_meta_path, 'r') as f:
            line = f.readline()
            while line:
                tmp = line.split(',')
                file2tag[tmp[0]] = int(tmp[1])
                line = f.readline()

    train_path = os.path.join(args.to_path, "train")
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    test_path = os.path.join(args.to_path, "test")
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    if args.opt == "random":
        random_split()
