//
// Created by Yanyu on 2022/7/14.
//

#ifndef FALIB_FA_H
#define FALIB_FA_H

#include "face_align.h"
#include "face_depth_map.h"

class FA{
public:
    FA(int alignSize=384, int faceSize=288, int maxAlignOffsetPixel=4, int maxKeyPointsNum=256, float normalizePercent=0.95f);
    virtual ~FA();

    float* getTransParams();
    // 因为要给python计算深度图，所以不能事先裁剪好，所以裁剪成256*256的工作在上层完成
    unsigned char* getAlignImage();
    unsigned char* getDepthMap();

    void cutFace(unsigned char* img, int height, int width, float* transParams);

    void fastAlign(unsigned char* img, int height, int width, int leftEyeX, int leftEyeY, int rightEyeX, int rightEyeY);
    void slowAlign(unsigned char* img, int height, int width, int* keyPointsX, int* keyPointsY, int keyPointsNum, int leftEyeId, int rightEyeId, float mergeWeight);
    void pixelInterpolation(float* vertices, int verticesNum);
protected:
    FaceAlign faceAlign;
    FaceDepthMap faceDepthMap;
    float normalizePercent;

    // fromX, fromY, toX, toY, scale, cosAngle, sinAngle
    float* outTransParams;
    unsigned char* outAlignImg;
    unsigned char* depthMap;
};

#endif //FALIB_FA_H
