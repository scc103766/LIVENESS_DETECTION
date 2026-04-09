import ctypes
import platform
import os
import numpy as np
import cv2
import time
from ctypes import c_void_p, c_float, c_bool, c_char, c_char_p, c_int

sysstr = platform.system()
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_path = curr_path + "/"

try:
    tg_infer = ctypes.windll.LoadLibrary(lib_path + 'tg_infer_dymc.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libtg_infer_dymc.so')
except OSError as e:
    lib_path = curr_path + "/../../lib_infer/build/"
    tg_infer = ctypes.windll.LoadLibrary(
        lib_path + 'Release/tg_infer_dymc.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libtg_infer_dymc.so')

tg_infer.createTGInfer.restype = c_void_p
tg_infer.infer.restype = c_bool
tg_infer.checkColor.restype = c_bool
tg_infer.getScore.restype = c_float


class TGInfer(object):
    def __init__(self, proto_path, model_path, height=256, width=256, n=3):
        self.height = height
        self.width = width
        self.n = n
        self.libtg_infer = c_void_p(tg_infer.createTGInfer(height, width, n, proto_path.encode(), model_path.encode()))

        self.i_color_list = (c_int * n)()

    def __del__(self):
        tg_infer.releaseTGInfer(self.libtg_infer)

    def predict(self, img_list, screen_color_list):
        img_cache = np.zeros([self.n, self.height, self.width, 3], dtype=np.uint8)
        for id, img in enumerate(img_list):
            img_cache[id, :, :, :] = img
        # .transpose((0, 3, 1, 2))
        if tg_infer.infer(self.libtg_infer, img_cache.ctypes.data_as(ctypes.c_char_p)):
            for i in range(self.n):
                self.i_color_list[i] = int(screen_color_list[i])
            if tg_infer.checkColor(self.libtg_infer, self.i_color_list):
                return float(tg_infer.getScore(self.libtg_infer))
            # 颜色验证失败
            return -1.0
        # 推理失败
        return None
