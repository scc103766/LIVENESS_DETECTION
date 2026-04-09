//
// Created by Yanyu on 2022/7/14.
//

#ifndef FALIB_NX_FA_EXPORT_H
#define FALIB_NX_FA_EXPORT_H

#ifdef WIN32 // or something like that...

#ifdef _USRDLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#else
#define DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 程序句柄
struct FA_Handle;

// 初始化参数
struct FA_Init_Parameter{
    // 图片的高
    int height;
    // 图片的宽
    int width;
    // 帧数量
    int frameNum;
    // 人脸画框的大小，默认384
    int alignSize;
    // 在人脸画框中，人脸的大小，默认208
    int faceSize;
    // 从人脸画框中，裁剪人脸的大小，默认256
    int cutSize;
    // 人脸对齐时抖动的像素，默认是4
    int maxAlignOffsetPixel;
    // 对齐时使用人脸关键点的数量
    int maxKeyPointsNum;
    // 左右眼在关键点的id
    int leftEyeId, rightEyeId;
    // 对照片的对比度使用排序法做归一化，拍摄使用的像素比例，默认是0.95。如果不想对裁剪的照片做对比度归一化，传入-1
    float normalizePercent;
    // 对齐时使用的融合系数，默认是1
    float mergeWeight;
};

/*
 * 初始化程序：
 * 返回值：
 *  0：成功
 *  1：左右眼id范围错误
 *  2：cutSize必须等于256
 * */
DLLEXPORT int FA_Initial(FA_Handle **ppHandle, const FA_Init_Parameter& initParameter);

// 获取版本号
DLLEXPORT const char* FA_GetVersion();

// 释放程序资源，返回0表示成功
DLLEXPORT int FA_Uninitial(FA_Handle *pHandle);

// 插入图片的参数
struct FA_Insert_Parameter{
    // 插入的一帧图片
    unsigned char *img;
    // 插入的图片id
    int frameId;
};

/*
 * 一帧一帧插入图片:
 * 返回值：
 *  0：成功
 *  1：frameId不对
 * */
DLLEXPORT int FA_InsertImage(FA_Handle *pHandle, FA_Insert_Parameter insertParameter);

/*
 * 获取当前插入的图片
 * */
DLLEXPORT unsigned char* FA_GetCurrentInsertImage(FA_Handle *pHandle, int frameId);

/*
 * 设置当前插入图片的人脸关键点
 *返回值：
 *  0：成功
 *  1：frameId不对
 * */
struct FA_KeyPoints_Parameter{
    int* keyPointsX;
    int* keyPointsY;
    int frameId;
};
DLLEXPORT int FA_SetCurrentInsertKeyPoints(FA_Handle *pHandle, const FA_KeyPoints_Parameter& keyPointsParameter);

/*
* 结束插入图片:
* 返回值：
*  0：成功
*  1：插入的图片不够
*  2：插入的关键点不够
* */
DLLEXPORT int FA_FinishInsertImage(FA_Handle *pHandle);

// 获得对齐的人脸图
DLLEXPORT unsigned char* FA_GetAlignFace(FA_Handle* pHandle);

#ifdef __cplusplus
}
#endif

#endif //FALIB_NX_FA_EXPORT_H
