import argparse
import os
from tqdm import tqdm
from libfa import FaceAlign
import cv2
import numpy as np
from predictor import PosPrediction
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--from_path', type=str, default="../data/choose", help='原始文件路径')
parser.add_argument('--to_path', type=str, default="../data/align", help="生成文件路径")
parser.add_argument('--align_size', type=int, default=384, help="对齐后图片的大小")
parser.add_argument('--face_size', type=int, default=208, help="对齐后人脸区域的大小")
parser.add_argument('--model_path', type=str, default="resources", help="模型位置")
parser.add_argument('--max_align_offset_pixel', type=int, default=4, help="为了找到最合适的对齐点，像素抖动的范围")
parser.add_argument('--normalize_percent', type=float, default=-1.0, help="亮度归一化参数，仅normalize_percent的像素参与归一化")
args = parser.parse_args()


def read_eye_pos_list(path):
    pos_list = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            pos_list.append(list(map(lambda a: int(a), line.split(','))))
            line = f.readline()
    return pos_list


def read_img_list(path):
    file_list = []
    if not os.path.exists(path):
        return None
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png"):
            tmp = file.split('.')
            file_list.append((int(tmp[0]), tmp[1]))
    file_list = sorted(file_list, key=lambda a: a[0])
    file_list = list(map(lambda a: "%d.%s" % (a[0], a[1]), file_list))
    frame_list = [cv2.imread(os.path.join(path, file)) for file in file_list]
    return frame_list


def back_trans(key_point_x, key_point_y, trans_pars):
    # 将对齐后的坐标还原回去
    from_x = trans_pars[0]
    from_y = trans_pars[1]
    to_x = trans_pars[2]
    to_y = trans_pars[3]
    scale = trans_pars[4]
    cos_angle = trans_pars[5]
    sin_angle = trans_pars[6]
    mov_x = from_x - to_x
    mov_y = from_y - to_y
    new_key_point_x, new_key_point_y = [], []
    for j, i in zip(key_point_x, key_point_y):
        f_offset_y = (i - to_y) * scale
        f_offset_x = (j - to_x) * scale
        f_i = sin_angle * f_offset_x + cos_angle * f_offset_y + to_y + mov_y
        f_j = cos_angle * f_offset_x - sin_angle * f_offset_y + to_x + mov_x
        new_key_point_x.append(int(f_j))
        new_key_point_y.append(int(f_i))
    return new_key_point_x, new_key_point_y


def draw_kps(img, kps, point_size=2):
    img = np.array(img, np.uint8)
    for i in range(kps.shape[0]):
        cv2.circle(img, (int(kps[i, 0]), int(kps[i, 1])), point_size, (0, 255, 0), -1)
    return img


def test_show(face_align, cropped_img, cropped_pos, face_kps):
    print(face_kps.shape)
    # 计算深度图
    all_vertices = np.reshape(cropped_pos, [256 * 256, -1])
    vertices = all_vertices[face_ind, :]
    depth = face_align.cal_depth_face(vertices)
    plt.imshow(draw_kps(np.where(depth > 0, depth / 255., cropped_img).copy(), face_kps))
    plt.axis('off')
    plt.show()


def main():
    face_align = FaceAlign(args.align_size, args.face_size, args.max_align_offset_pixel,
                           normalize_percent=args.normalize_percent)
    file_list = [file for file in os.listdir(args.from_path) if file.endswith(".txt")]
    for file in tqdm(file_list):
        if os.path.exists(os.path.join(args.to_path, file)):
            continue
        pos_list = read_eye_pos_list(os.path.join(args.from_path, file))
        img_list = read_img_list(os.path.join(args.from_path, file.replace(".txt", "")))
        if img_list is None:
            continue
        assert len(pos_list) == len(img_list)

        # 填写缺失的双目位置信息
        pos_id = -1
        for id, pos in enumerate(pos_list):
            if pos[1] > 0:
                pos_id = id
                break
        if pos_id == -1:
            # 找不到双眼坐标
            continue
        best_pos = pos_list[pos_id]
        for pos in pos_list:
            if pos[1] <= 0:
                pos[1], pos[2], pos[3], pos[4] = best_pos[1], best_pos[2], best_pos[3], best_pos[4]

        key_point_x_list, key_point_y_list = [], []
        for index, (img, pos) in enumerate(zip(img_list, pos_list)):
            fast_align_img, trans_pars = face_align.fast_align(img, pos[1], pos[2], pos[3], pos[4])

            # plt.imshow(fast_align_img)
            # plt.axis('off')
            # plt.show()

            cropped_img = cv2.cvtColor(cv2.resize(fast_align_img, (256, 256)), cv2.COLOR_BGR2RGB) / 255.
            cropped_pos = pos_predictor.predict(cropped_img)
            # 左眼：37、38、40、41，右眼：43、44、46、47
            face_kps = cropped_pos[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]

            key_point_x = face_kps[:, 0].tolist()
            key_point_y = face_kps[:, 1].tolist()
            left_eye_id = len(key_point_x)
            right_eye_id = left_eye_id + 1
            # 增加左右眼的位置
            left_x = key_point_x[37] + key_point_x[38] + key_point_x[40] + key_point_x[41]
            left_y = key_point_y[37] + key_point_y[38] + key_point_y[40] + key_point_y[41]
            right_x = key_point_x[43] + key_point_x[44] + key_point_x[46] + key_point_x[47]
            right_y = key_point_y[43] + key_point_y[44] + key_point_y[46] + key_point_y[47]

            key_point_x.extend([left_x * 0.25, right_x * 0.25])
            key_point_y.extend([left_y * 0.25, right_y * 0.25])

            # 测试效果是否准确
            # test_show(face_align, cropped_img, cropped_pos, np.concatenate([np.array(key_point_x).reshape([-1,1]), np.array(key_point_y).reshape([-1,1])], axis=1))

            # 将人脸关键点还原回去-----------
            def resize256to_align(data_list):
                return list(map(lambda d: d * args.align_size / 256.0, data_list))

            key_point_x, key_point_y = back_trans(resize256to_align(key_point_x), resize256to_align(key_point_y),
                                                  trans_pars)
            key_point_x_list.append(key_point_x)
            key_point_y_list.append(key_point_y)

        # 开始写入参数
        with open(os.path.join(args.to_path, file.replace(".txt", "_param.txt")), 'w') as f:
            pre_color = -1
            for id, (source_line, key_point_x, key_point_y) in enumerate(zip(pos_list, key_point_x_list, key_point_y_list)):
                color = source_line[-1]
                if color != pre_color:
                    pre_color = color
                    if color == 0 or id == 0:
                        state = 0
                    else:
                        state = 1
                else:
                    if color == 16777215:
                        state = 2
                    elif color == 0:
                        state = 3
                    else:
                        state = 4
                new_line_list = [state, color]
                new_line_list.extend(key_point_x)
                new_line_list.extend(key_point_y)
                new_line = ",".join(list(map(lambda a: str(a), new_line_list)))
                f.write(new_line + "\n")


if __name__ == '__main__':
    # 网络推理
    pos_predictor = PosPrediction(256, 256)
    pos_predictor.restore(os.path.join(args.model_path, "net-data", "256_256_resfcn256_weight"))
    # 这里是标准面部空间的所有点，也是配置死的
    face_ind = np.loadtxt(os.path.join(args.model_path, "uv-data", "face_ind.txt")).astype(np.int32)
    # 人脸关键点
    uv_kpt_ind = np.loadtxt(os.path.join(args.model_path, "uv-data", "uv_kpt_ind.txt")).astype(np.int32)
    main()
