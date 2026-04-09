//
// Created by Yanyu on 2022/7/14.
//
#include <algorithm>
#include <cmath>
#include "img_proc.h"
#include "face_align.h"
#include <string.h>


#define ALIGN_CACHE_SIZE 64

template<class T>
T clamp(T x, T min, T max) {
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

FaceAlign::FaceAlign(int alignSize, int faceSize, int maxAlignOffsetPixel, int maxKeyPointsNum) : alignSize(alignSize),
                                                                                                  faceSize(faceSize),
                                                                                                  maxAlignOffsetPixel(
                                                                                                          maxAlignOffsetPixel),
                                                                                                  maxKeyPointsNum(
                                                                                                          maxKeyPointsNum) {
    alignImageCache = new unsigned char[alignSize * alignSize * 3];

    anchorKeyPointsXCache = new int[maxKeyPointsNum * 2];
    anchorKeyPointsYCache = new int[maxKeyPointsNum * 2];
    // 上一张特征图+当前特征图
    anchorColorCache = new unsigned char[ALIGN_CACHE_SIZE * ALIGN_CACHE_SIZE * 3 * 2];

    sobelCache = new float[ALIGN_CACHE_SIZE * ALIGN_CACHE_SIZE * 3];
    anchorSobelCache = new unsigned char[ALIGN_CACHE_SIZE * ALIGN_CACHE_SIZE * 3 * 2];

    int offsetSize = maxAlignOffsetPixel * 2 - 1;
    flag = new int[offsetSize * offsetSize * offsetSize * offsetSize];
}

FaceAlign::~FaceAlign() {
    delete[]alignImageCache;
    delete[]anchorKeyPointsXCache;
    delete[]anchorKeyPointsYCache;
    delete[]anchorColorCache;
    delete[]sobelCache;
    delete[]anchorSobelCache;
}

inline void
caluateTransParams(float *outTransParams, int alignSize, int faceSize, int leftEyeX, int leftEyeY, int rightEyeX,
                   int rightEyeY, int offsetX = 0, int offsetY = 0, int offsetScale = 0, int offsetAngle = 0) {
    // 两眼之间的点在目标图的位置
    int toX = (alignSize >> 1);
    int toY = (faceSize >> 2) + ((alignSize - faceSize) >> 1);
    float toDis = (faceSize >> 1);
    // 两眼之间的点在原图的位置
    int fromX = ((leftEyeX + rightEyeX) >> 1) + offsetX;
    int fromY = ((leftEyeY + rightEyeY) >> 1) + offsetY;
    int disX = rightEyeX - leftEyeX;
    int disY = rightEyeY - leftEyeY;
    float fromDis = sqrtf(disX * disX + disY * disY);
    // 计算缩放经过偏移后的缩放
    float scale = (fromDis + offsetScale) / toDis;
    // 计算经过角度偏移后的夹角
    auto tanTheta = disY / (disX + 1e-5);
    auto theta = atan(tanTheta) + offsetAngle * 2.f / fromDis;
    float cosAngle = cos(theta);
    float sinAngle = sin(theta);
    // 输出对齐后的映射关系，也可以用矩阵实现
    outTransParams[0] = fromX;
    outTransParams[1] = fromY;
    outTransParams[2] = toX;
    outTransParams[3] = toY;
    outTransParams[4] = scale;
    outTransParams[5] = cosAngle;
    outTransParams[6] = sinAngle;
}

inline void
calTransKeyPoint(int *toKeyPointX, int *toKeyPointY, int *fromKeyPointX, int *fromKeyPointY, int keyPointNum,
                 int leftEyeId, int rightEyeId, int offsetX = 0, int offsetY = 0, int offsetScale = 0,
                 int offsetAngle = 0) {
    // 两眼之间的点在原图的位置
    int fromX = (fromKeyPointX[leftEyeId] + fromKeyPointX[rightEyeId]) >> 1;
    int fromY = (fromKeyPointY[leftEyeId] + fromKeyPointY[rightEyeId]) >> 1;
    int disX = fromKeyPointX[rightEyeId] - fromKeyPointX[leftEyeId];
    int disY = fromKeyPointY[rightEyeId] - fromKeyPointY[leftEyeId];
    float fromDis = sqrtf(disX * disX + disY * disY);
    // 计算缩放经过偏移后的缩放
    float scale = (fromDis + offsetScale) / fromDis;
    // 计算偏移弧度
    auto theta = offsetAngle * 2.f / fromDis;
    float cosAngle = cos(theta);
    float sinAngle = sin(theta);
    for (int i = 0; i < keyPointNum; ++i) {
        auto offsetY = (fromKeyPointY[i] - fromY) * scale;
        auto offsetX = (fromKeyPointX[i] - fromX) * scale;
        toKeyPointY[i] = sinAngle * offsetX + cosAngle * offsetY + fromY;
        toKeyPointX[i] = cosAngle * offsetX - sinAngle * offsetY + fromX;
    }
}

void FaceAlign::fastAlign(unsigned char *img, int height, int width, int leftEyeX, int leftEyeY, int rightEyeX,
                          int rightEyeY, float *outTransParams, unsigned char *outAlignImg) {
    caluateTransParams(outTransParams, alignSize, faceSize, leftEyeX, leftEyeY, rightEyeX, rightEyeY);
    if (outAlignImg != nullptr) {
        cutFace(img, height, width, outTransParams, outAlignImg);
    }
}


void FaceAlign::slowAlign(unsigned char *img, int height, int width, int *keyPointsX, int *keyPointsY, int keyPointsNum,
                          int leftEyeId, int rightEyeId, float mergeWeight, float *outTransParams,
                          unsigned char *outAlignImg) {
    if (mergeWeight < 0.f) {
        // 覆盖式填写锚点和锚图
        // 先对齐到缓存
        fastAlign(img, height, width, keyPointsX[leftEyeId], keyPointsY[leftEyeId], keyPointsX[rightEyeId],
                  keyPointsY[rightEyeId], outTransParams, alignImageCache);
        if (outAlignImg != nullptr)
            // 复制到输出
            memcpy(outAlignImg, alignImageCache, alignSize * alignSize * 3 * sizeof(unsigned char));
        // 直接覆盖锚点
        memcpy(anchorKeyPointsXCache, keyPointsX, keyPointsNum * sizeof(int));
        memcpy(anchorKeyPointsYCache, keyPointsY, keyPointsNum * sizeof(int));
        // 直接覆盖锚图
        copyToAnchorImage(alignImageCache, anchorColorCache, anchorSobelCache);
    } else {
        // 融合式填写锚点和锚图
        auto anchorOffset = ALIGN_CACHE_SIZE * ALIGN_CACHE_SIZE * 3;
        // 先仅仅根据关键点计算最优的偏移
        int offsetKeyPoint[] = {0, 0, 0, 0};
        float minLoss = 1e32;
        while (true) {
            int minI = -1;
            int minJ = 0;
            for (int i = 0; i < 4; ++i) {
                for (int j = -1; j < 2; j += 2) {
                    offsetKeyPoint[i] += j;
                    calTransKeyPoint(anchorKeyPointsXCache + maxKeyPointsNum, anchorKeyPointsYCache + maxKeyPointsNum,
                                     keyPointsX, keyPointsY, keyPointsNum, leftEyeId, rightEyeId, offsetKeyPoint[0],
                                     offsetKeyPoint[1], offsetKeyPoint[2], offsetKeyPoint[3]);
                    auto curLoss = calKeyPointLoss(keyPointsNum);
                    if (curLoss < minLoss) {
                        minLoss = curLoss;
                        minI = i;
                        minJ = j;
                    }
                    offsetKeyPoint[i] -= j;
                }
            }
            if (minI >= 0) {
                offsetKeyPoint[minI] = minJ;
            } else {
                break;
            }
        }

        // 再计算像素级别的偏移
        int offsetPixel[] = {0, 0, 0, 0};
        int offsetSize = maxAlignOffsetPixel * 2 - 1;
        memset(flag, 0, offsetSize * offsetSize * offsetSize * offsetSize * sizeof(int));
        minLoss = 1e32;
        while (true) {
            int minI = -1;
            int minJ = 0;
            for (int i = 0; i < 4; ++i) {
                for (int j = -1; j < 2; j += 2) {
                    int flagId = offsetPixel[0] * offsetSize * offsetSize * offsetSize +
                                 offsetPixel[1] * offsetSize * offsetSize + offsetPixel[2] * offsetSize +
                                 offsetPixel[3];
                    if (offsetPixel[i] + j <= -maxAlignOffsetPixel || offsetPixel[i] + j >= maxAlignOffsetPixel ||
                        flag[flagId] > 0)
                        continue;
                    flag[flagId] = 1;

                    offsetPixel[i] += j;
                    caluateTransParams(outTransParams, alignSize, faceSize, keyPointsX[leftEyeId],
                                       keyPointsY[leftEyeId], keyPointsX[rightEyeId], keyPointsY[rightEyeId],
                                       offsetKeyPoint[0] + offsetPixel[0], offsetKeyPoint[1] + offsetPixel[1],
                                       offsetKeyPoint[2] + offsetPixel[2], offsetKeyPoint[3] + offsetPixel[3]);
                    cutFace(img, height, width, outTransParams, alignImageCache);

                    copyToAnchorImage(alignImageCache, anchorColorCache + anchorOffset,
                                      anchorSobelCache + anchorOffset);
                    auto curLoss = calColorLoss();

                    if (curLoss < minLoss) {
                        minLoss = curLoss;
                        minI = i;
                        minJ = j;
                    }
                    offsetPixel[i] -= j;
                }
            }
            if (minI >= 0) {
                offsetPixel[minI] = minJ;
            } else {
                break;
            }
        }

        // 找到最佳偏移
        caluateTransParams(outTransParams, alignSize, faceSize, keyPointsX[leftEyeId], keyPointsY[leftEyeId],
                           keyPointsX[rightEyeId], keyPointsY[rightEyeId], offsetKeyPoint[0] + offsetPixel[0],
                           offsetKeyPoint[1] + offsetPixel[1], offsetKeyPoint[2] + offsetPixel[2],
                           offsetKeyPoint[3] + offsetPixel[3]);
        cutFace(img, height, width, outTransParams, alignImageCache);
        // 输出
        if (outAlignImg != nullptr) {
            memcpy(outAlignImg, alignImageCache, alignSize * alignSize * 3 * sizeof(unsigned char));
        }
        // 锚点和锚图融合
        calTransKeyPoint(anchorKeyPointsXCache + maxKeyPointsNum, anchorKeyPointsYCache + maxKeyPointsNum, keyPointsX,
                         keyPointsY, keyPointsNum, leftEyeId, rightEyeId, offsetKeyPoint[0] + offsetPixel[0],
                         offsetKeyPoint[1] + offsetPixel[1], offsetKeyPoint[2] + offsetPixel[2],
                         offsetKeyPoint[3] + offsetPixel[3]);
        copyToAnchorImage(alignImageCache, anchorColorCache + anchorOffset, anchorSobelCache + anchorOffset);
        mergeAnchor(mergeWeight);
    }
}


void FaceAlign::cutFace(unsigned char *img, int height, int width, float *transParams, unsigned char *outAlignImg) {
    auto fromX = transParams[0];
    auto fromY = transParams[1];
    auto toX = transParams[2];
    auto toY = transParams[3];
    auto scale = transParams[4];
    auto cosAngle = transParams[5];
    auto sinAngle = transParams[6];
    float movX = fromX - toX;
    float movY = fromY - toY;
    for (int i = 0; i < alignSize; ++i) {
        // 眉心偏移
        float fOffsetY = (i - toY) * scale;
        for (int j = 0; j < alignSize; ++j) {
            float fOffsetX = (j - toX) * scale;
            float fi = clamp<float>(sinAngle * fOffsetX + cosAngle * fOffsetY + toY + movY, 2.f, height - 2.f);
            float fj = clamp<float>(cosAngle * fOffsetX - sinAngle * fOffsetY + toX + movX, 2.f, width - 2.f);
            ImgProc::bilinera(img, width, fi, fj, outAlignImg + (i * alignSize + j) * 3, 3);
        }
    }
}

void FaceAlign::copyToAnchorImage(unsigned char *alignImage, unsigned char *toAnchorColorCache,
                                  unsigned char *toAnchorSobelCache) {
    float scale = faceSize / (float) ALIGN_CACHE_SIZE;
    int offset = ((alignSize - faceSize) >> 1);
    for (int i = 0; i < ALIGN_CACHE_SIZE; ++i) {
        float fi = (i + 0.5f) * scale + offset - 0.5f;
        for (int j = 0; j < ALIGN_CACHE_SIZE; ++j) {
            float fj = (j + 0.5f) * scale + offset - 0.5f;
            ImgProc::bilinera(alignImage, alignSize, fi, fj, toAnchorColorCache + (i * ALIGN_CACHE_SIZE + j) * 3, 3);
        }
    }
    ImgProc::sobel(toAnchorColorCache, ALIGN_CACHE_SIZE, ALIGN_CACHE_SIZE, sobelCache);
    ImgProc::float2uchar(sobelCache, ALIGN_CACHE_SIZE, ALIGN_CACHE_SIZE, toAnchorSobelCache);
}

float FaceAlign::calKeyPointLoss(int keyPointNum) {
    float loss = 0;
    for (int i = 0; i < keyPointNum; ++i) {
        auto disX = anchorKeyPointsXCache[i] - anchorKeyPointsXCache[maxKeyPointsNum + 1];
        auto disY = anchorKeyPointsYCache[i] - anchorKeyPointsYCache[maxKeyPointsNum + 1];
        loss += disX * disX + disY * disY;
    }
    return loss;
}

float FaceAlign::calColorLoss() {
    return 0;
}

void FaceAlign::mergeAnchor(float mergeWeight) {
    int pixelNum = ALIGN_CACHE_SIZE * ALIGN_CACHE_SIZE * 3;
    for (int i = 0; i < pixelNum; ++i) {
        anchorColorCache[i] = (unsigned char) (anchorColorCache[i] * (1 - mergeWeight) +
                                               anchorColorCache[i + pixelNum] * mergeWeight);
        anchorSobelCache[i] = (unsigned char) (anchorSobelCache[i] * (1 - mergeWeight) +
                                               anchorSobelCache[i + pixelNum] * mergeWeight);
    }
    for (int i = 0; i < maxKeyPointsNum; ++i) {
        anchorKeyPointsXCache[i] = (int) (anchorKeyPointsXCache[i] * (1 - mergeWeight) +
                                          anchorKeyPointsXCache[i + maxKeyPointsNum] * mergeWeight);
        anchorKeyPointsYCache[i] = (int) (anchorKeyPointsYCache[i] * (1 - mergeWeight) +
                                          anchorKeyPointsYCache[i + maxKeyPointsNum] * mergeWeight);
    }
}