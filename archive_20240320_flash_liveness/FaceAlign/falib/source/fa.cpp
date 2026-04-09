//
// Created by Yanyu on 2022/7/14.
//

#include <string.h>
#include <iostream>
#include "fa.h"
#include "img_proc.h"

FA::FA(int alignSize, int faceSize, int maxAlignOffsetPixel, int maxKeyPointsNum, float normalizePercent):faceAlign(alignSize, faceSize, maxAlignOffsetPixel, maxKeyPointsNum), faceDepthMap(256, 256), normalizePercent(normalizePercent) {
    outTransParams = new float [7];
    // 前面：alignSize * alignSize * 3存人脸图，后面：alignSize * alignSize * 1作为缓存
    outAlignImg = new unsigned char [alignSize * alignSize * 4];
    depthMap = new unsigned char [256 * 256];

}

FA::~FA() {
    delete []outTransParams;
    delete []outAlignImg;
    delete []depthMap;
}

float* FA::getTransParams() {return outTransParams;}
unsigned char* FA::getAlignImage() {
    return outAlignImg;
}
unsigned char* FA::getDepthMap() {
    return depthMap;
}

void FA::cutFace(unsigned char *img, int height, int width, float *transParams) {
    faceAlign.cutFace(img, height, width, transParams, outAlignImg);
}

void FA::fastAlign(unsigned char *img, int height, int width, int leftEyeX, int leftEyeY, int rightEyeX,
                   int rightEyeY) {
    faceAlign.fastAlign(img, height, width, leftEyeX, leftEyeY, rightEyeX, rightEyeY, outTransParams, outAlignImg);
    if(normalizePercent > 0){
        auto cache = outAlignImg + (faceAlign.getAlignSize() * faceAlign.getAlignSize() * 3);
        ImgProc::normalize(outAlignImg, cache, faceAlign.getAlignSize(), faceAlign.getAlignSize(), 3, normalizePercent);
    }
}

void FA::slowAlign(unsigned char *img, int height, int width, int *keyPointsX, int *keyPointsY, int keyPointsNum,
                   int leftEyeId, int rightEyeId, float mergeWeight) {
    faceAlign.slowAlign(img, height, width, keyPointsX, keyPointsY, keyPointsNum, leftEyeId, rightEyeId, mergeWeight, outTransParams, outAlignImg);
    if(normalizePercent > 0){
        auto cache = outAlignImg + (faceAlign.getAlignSize() * faceAlign.getAlignSize() * 3);
        ImgProc::normalize(outAlignImg, cache, faceAlign.getAlignSize(), faceAlign.getAlignSize(), 3, normalizePercent);
    }
}

void FA::pixelInterpolation(float* vertices, int verticesNum){
    faceDepthMap.pixelInterpolation(vertices, verticesNum, depthMap);
}