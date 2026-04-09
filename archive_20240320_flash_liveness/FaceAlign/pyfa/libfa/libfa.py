import ctypes
import platform
import os
import numpy as np
from ctypes import c_void_p, c_float

sysstr = platform.system()
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
lib_path = curr_path + "/"

try:
    fa = ctypes.windll.LoadLibrary(lib_path + 'fa_dymc.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libfa_dymc.so')
except OSError as e:
    lib_path = curr_path + "/../../falib/build/"
    fa = ctypes.windll.LoadLibrary(
        lib_path + 'Release/fa_dymc.dll') if sysstr == "Windows" else ctypes.cdll.LoadLibrary(
        lib_path + 'libfa_dymc.so')

fa.createLibFA.restype = c_void_p


class FaceAlign(object):
    def __init__(self, align_size=384, face_size=208, max_align_offset_pixel=4, max_key_points_num=128,
                 normalize_percent=-1.0):
        # 做人脸对齐后，会将图片切割成align_size*align_size
        self.align_size = align_size
        # 在align_size中心，选择face_size*face_size的方框，将眼睛放在方框的上1/4位置，左右眼分别放在方框的左1/4和右1/4位置
        self.face_size = face_size
        self.libfa = c_void_p(
            fa.createLibFA(align_size, face_size, max_align_offset_pixel, max_key_points_num, c_float(normalize_percent)))

        self.o_trans_type = ctypes.c_float * 7
        fa.getTransParams.restype = ctypes.POINTER(ctypes.c_float * 7)

        self.o_img_type = ctypes.c_uint8 * (align_size * align_size * 3)
        fa.getAlignImage.restype = ctypes.POINTER(ctypes.c_uint8 * (align_size * align_size * 3))

        self.d_img_type = ctypes.c_uint8 * (256 * 256 * 1)
        fa.getDepthMap.restype = ctypes.POINTER(ctypes.c_uint8 * (256 * 256 * 1))

    def __del__(self):
        fa.releaseLibFA(self.libfa)

    def fast_align(self, img, left_eye_x, left_eye_y, right_eye_x, right_eye_y):
        height, width = img.shape[:2]
        fa.fastAlign(self.libfa, img.ctypes.data_as(ctypes.c_char_p), height, width, left_eye_x, left_eye_y,
                     right_eye_x, right_eye_y)

        trans_res = fa.getTransParams(self.libfa)
        trans_pointer = ctypes.cast(trans_res, ctypes.POINTER(self.o_trans_type))
        trans_res = np.frombuffer(trans_pointer.contents, dtype=np.float32).reshape([7])

        img_res = fa.getAlignImage(self.libfa)
        array_pointer = ctypes.cast(img_res, ctypes.POINTER(self.o_img_type))
        img_res = np.frombuffer(array_pointer.contents, dtype=np.uint8).reshape([self.align_size, self.align_size, 3])
        return img_res.copy(), trans_res.copy()

    def slow_align(self, img, key_point_x_list, key_point_y_list, left_eye_id, right_eye_id, merge_weight=-1.):
        height, width = img.shape[:2]
        key_point_x = np.array(key_point_x_list, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        key_point_y = np.array(key_point_y_list, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        key_point_num = len(key_point_x_list)
        fa.slowAlign(self.libfa, img.ctypes.data_as(ctypes.c_char_p), height, width, key_point_x, key_point_y,
                     key_point_num, left_eye_id, right_eye_id, c_float(merge_weight))

        trans_res = fa.getTransParams(self.libfa)
        trans_pointer = ctypes.cast(trans_res, ctypes.POINTER(self.o_trans_type))
        trans_res = np.frombuffer(trans_pointer.contents, dtype=np.float32).reshape([7])

        img_res = fa.getAlignImage(self.libfa)
        array_pointer = ctypes.cast(img_res, ctypes.POINTER(self.o_img_type))
        img_res = np.frombuffer(array_pointer.contents, dtype=np.uint8).reshape([self.align_size, self.align_size, 3])
        return img_res.copy(), trans_res.copy()

    def cut_face(self, img, trans):
        height, width = img.shape[:2]
        fa.cutFace(self.libfa, img.ctypes.data_as(ctypes.c_char_p), height, width,
                   trans.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        img_res = fa.getAlignImage(self.libfa)
        array_pointer = ctypes.cast(img_res, ctypes.POINTER(self.o_img_type))
        img_res = np.frombuffer(array_pointer.contents, dtype=np.uint8).reshape([self.align_size, self.align_size, 3])
        return img_res.copy()

    def cal_depth_face(self, vertices):
        vertices = np.reshape(np.array(vertices, dtype=np.float32), [-1, 3])
        fa.pixelInterpolation(self.libfa, vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(vertices))
        res = fa.getDepthMap(self.libfa)
        array_pointer = ctypes.cast(res, ctypes.POINTER(self.d_img_type))
        return (np.frombuffer(array_pointer.contents, dtype=np.uint8).reshape([256, 256, 1]) * np.ones([1, 1, 3],
                                                                                                       dtype=np.uint8)).copy()
