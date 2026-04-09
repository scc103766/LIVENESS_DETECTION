import ctypes
import platform
import os
import numpy as np
import cv2
from ctypes import c_void_p, c_float, c_bool, c_int

sysstr = platform.system()
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_path = curr_path + "/"

try:
    tg = ctypes.windll.LoadLibrary(lib_path + 'tg_dymc.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libtg_dymc.so')
except OSError as e:
    lib_path = curr_path + "/../../libtg/build/"
    tg = ctypes.windll.LoadLibrary(
        lib_path + 'Release/tg_dymc.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libtg_dymc.so')

tg.createLibtg.restype = c_void_p
tg.isInsertImg.restype = c_bool


class TG(object):
    def __init__(self, mask_path, align_size=480, n=5, cut_size=288, normalize_size=256, normalize_percent=0.9):
        mask = np.array(cv2.imread(mask_path), dtype=np.uint8)
        self.img_height = mask.shape[0]
        self.img_width = mask.shape[1]
        self.align_size = align_size
        self.n = n
        self.libtg = c_void_p(
            tg.createLibtg(self.img_height, self.img_width, mask.ctypes.data_as(ctypes.c_char_p), align_size, n, c_float(normalize_percent)))

        self.i_color_list = (c_int * n)()

        self.normal_cue_size = cut_size
        self.normalize_size = normalize_size
        self.n_img_type = ctypes.c_uint8 * (cut_size * cut_size * 3 * (n - 2) * 2)
        tg.calNormalCuesMap.restype = ctypes.POINTER(
            ctypes.c_uint8 * (cut_size * cut_size * 3 * (n - 2) * 2))

    def __del__(self):
        tg.releaseLibtg(self.libtg)

    def is_insert_image(self, img, screen_color, state, step=2):
        img = np.array(img, dtype=np.uint8)
        return tg.isInsertImg(self.libtg, img.ctypes.data_as(ctypes.c_char_p), screen_color, state, step)

    def cal_normal_cues_map(self, img, screen_color_list):
        for i in range(self.n):
            self.i_color_list[i] = screen_color_list[i]
        res = tg.calNormalCuesMap(self.libtg, self.i_color_list, img.ctypes.data_as(ctypes.c_char_p),
                                  self.normal_cue_size, self.normalize_size)
        array_pointer = ctypes.cast(res, ctypes.POINTER(self.n_img_type))
        return np.frombuffer(array_pointer.contents, dtype=np.uint8).reshape(
            [(self.n - 2) * 2 * self.normal_cue_size, self.normal_cue_size, 3])
