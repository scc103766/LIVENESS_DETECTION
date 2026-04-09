//
// Created by Yanyu on 2022/7/14.
//

#include "nx_fa_export.h"
#include "fa.h"
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;

struct FA_Handle{
    FA* fa;
    FA_Init_Parameter initParameter;

    // 帮助客户端缓存插入的图片
    unsigned char* insertImages;
    // 记录缓存图的编码和旋转
    int* encodes;
    int* rotates;

    // 帮助客户端缓存关键点检测的结果
    int *keyPointsX;
    int *keyPointsY;

    // 帮助客户端缓存对齐的图片
    unsigned char* alignImages;

    int maxImgFrameId;
    int maxKeyPointFrameId;
};

inline static void trans(int rotate, int height, int width, int i, int j, int& toI, int& toJ){
    switch (rotate) {
        case 90:
            toI = j;
            toJ = height - i - 1;
            break;
        case 180:
            toI = height - i - 1;
            toJ = width - j - 1;
            break;
        case 270:
            toI = width - j - 1;
            toJ = i;
            break;
        default:
            toI = i;
            toJ = j;
    }
}

inline static void decodeYUV420SP(unsigned char* yuv420sp, int width, int height, unsigned char *bgrBuf, int rotate)
{
    int frameSize = width * height;
    int toHeight = (rotate == 90 || rotate == 270) ? width : height;
    int toWidth = (rotate == 90 || rotate == 270) ? height : width;

    int i = 0, j = 0, y = 0;
    int toI = 0, toJ = 0;
    int yp = 0, uvp = 0, u = 0, v = 0;
    int y1192 = 0, r = 0, g = 0, b = 0;

    int bgrp = 0;
    for (i = 0; i < height; ++i){
        for (j = 0; j < width; ++j){
            trans(rotate, height, width, i, j, toI, toJ);
            yp = toI * toWidth + toJ;
            uvp = frameSize + (toI >> 1) * toWidth + (toJ & ~1);

            y = (0xff & ((int) yuv420sp[yp])) - 16;
            if (y < 0) y = 0;
            v = (0xff & yuv420sp[uvp]) - 128;
            u = (0xff & yuv420sp[uvp+1]) - 128;
            y1192 = 1192 * y;
            r = (y1192 + 1634 * v);
            g = (y1192 - 833 * v - 400 * u);
            b = (y1192 + 2066 * u);

            if (r < 0) r = 0; else if (r > 262143) r = 262143;
            if (g < 0) g = 0; else if (g > 262143) g = 262143;
            if (b < 0) b = 0; else if (b > 262143) b = 262143;

            bgrBuf[bgrp * 3 + 2] = (unsigned char)(r >> 10);
            bgrBuf[bgrp * 3 + 1] = (unsigned char)(g >> 10);
            bgrBuf[bgrp * 3 + 0] = (unsigned char)(b >> 10);
            ++bgrp;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

int FA_Initial(FA_Handle **ppHandle, const FA_Init_Parameter& initParameter){
    if(initParameter.cutSize > initParameter.alignSize)
        return 2;
    if(initParameter.leftEyeId >= initParameter.maxKeyPointsNum || initParameter.rightEyeId >= initParameter.maxKeyPointsNum)
        return 1;
    (*ppHandle) = new FA_Handle();
    auto pHandle = (*ppHandle);
    pHandle->initParameter = initParameter;
    pHandle->fa = new FA(initParameter.alignSize, initParameter.faceSize, initParameter.maxAlignOffsetPixel, initParameter.maxKeyPointsNum, initParameter.normalizePercent);

    // 流出最后一张图做缓存
    pHandle->insertImages = new unsigned char[initParameter.height * initParameter.width * 3 * (initParameter.frameNum + 1)];
    pHandle->encodes = new int[initParameter.frameNum];
    pHandle->rotates = new int[initParameter.frameNum];
    pHandle->keyPointsX = new int[initParameter.maxKeyPointsNum * initParameter.frameNum];
    pHandle->keyPointsY = new int[initParameter.maxKeyPointsNum * initParameter.frameNum];
    pHandle->alignImages = new unsigned char[initParameter.cutSize * initParameter.cutSize * 3 * initParameter.frameNum];

    pHandle->maxImgFrameId = 0;
    pHandle->maxKeyPointFrameId = 0;
    return 0;
}

const char *TG_GetVersion() { return "1.0.0.0"; }

int FA_Uninitial(FA_Handle *pHandle){
    delete pHandle->fa;
    delete[] pHandle->insertImages;
    delete[] pHandle->encodes;
    delete[] pHandle->rotates;
    delete[] pHandle->keyPointsX;
    delete[] pHandle->keyPointsY;
    delete[] pHandle->alignImages;
    return 0;
}

int FA_InsertImage(FA_Handle *pHandle, FA_Insert_Parameter insertParameter){
    if(insertParameter.frameId >= pHandle->initParameter.frameNum || insertParameter.frameId < 0)
        return 1;
    if(insertParameter.frameId > pHandle->maxImgFrameId)
        pHandle->maxImgFrameId = insertParameter.frameId;
    int imgSize = pHandle->initParameter.height * pHandle->initParameter.width * 3;
    pHandle->encodes[insertParameter.frameId] = insertParameter.encode;
    pHandle->rotates[insertParameter.frameId] = insertParameter.rotate;
    if(insertParameter.encode == 0){
        memcpy(pHandle->insertImages + imgSize * insertParameter.frameId, insertParameter.img, imgSize * sizeof(unsigned char ));
    } else if(insertParameter.encode == 1){
        int cSize = pHandle->initParameter.height * pHandle->initParameter.width + (pHandle->initParameter.height >> 1) * pHandle->initParameter.width;
        memcpy(pHandle->insertImages + imgSize * insertParameter.frameId, insertParameter.img, cSize * sizeof(unsigned char ));
    } else {
        return 2;
    }

    return 0;
}

unsigned char* FA_GetCurrentInsertImage(FA_Handle *pHandle, int frameId){
    if(frameId >= pHandle->initParameter.frameNum)
        frameId = pHandle->initParameter.frameNum - 1;
    else if(frameId < 0)
        frameId = 0;

    int imgSize = pHandle->initParameter.height * pHandle->initParameter.width * 3;
    auto img = pHandle->insertImages + imgSize * frameId;

    if(pHandle->encodes[frameId] == 1){
        // 当前图片编码是yuv420，需要转一下
        auto cache = pHandle->insertImages + imgSize * pHandle->initParameter.frameNum;
        decodeYUV420SP(img, pHandle->initParameter.width, pHandle->initParameter.height, cache, pHandle->rotates[frameId]);
        memcpy(img, cache, imgSize * sizeof(unsigned char ));
        pHandle->encodes[frameId] = 0;
        pHandle->rotates[frameId] = 0;
    }

    return img;
}

int FA_SetCurrentInsertKeyPoints(FA_Handle *pHandle, const FA_KeyPoints_Parameter& keyPointsParameter){
    if(keyPointsParameter.frameId >= pHandle->initParameter.frameNum || keyPointsParameter.frameId < 0)
        return 1;
    if(keyPointsParameter.frameId > pHandle->maxKeyPointFrameId)
        pHandle->maxKeyPointFrameId = keyPointsParameter.frameId;
    memcpy(pHandle->keyPointsX + pHandle->initParameter.maxKeyPointsNum * keyPointsParameter.frameId, keyPointsParameter.keyPointsX, pHandle->initParameter.maxKeyPointsNum * sizeof(int));
    memcpy(pHandle->keyPointsY + pHandle->initParameter.maxKeyPointsNum * keyPointsParameter.frameId, keyPointsParameter.keyPointsY, pHandle->initParameter.maxKeyPointsNum * sizeof(int));
    return 0;
}

int FA_FinishInsertImage(FA_Handle *pHandle){
    if(pHandle->maxImgFrameId != pHandle->initParameter.frameNum-1)
        return 1;
    if(pHandle->maxKeyPointFrameId != pHandle->initParameter.frameNum-1)
        return 2;
    pHandle->maxImgFrameId = 0;
    pHandle->maxKeyPointFrameId = 0;
    for(int frameId=0; frameId < pHandle->initParameter.frameNum; ++frameId){
        float mergeWeight = (frameId == 0 ? -1.f : pHandle->initParameter.mergeWeight);
        auto img = FA_GetCurrentInsertImage(pHandle, frameId);
        auto keyPointsX = pHandle->keyPointsX + pHandle->initParameter.maxKeyPointsNum * frameId;
        auto keyPointsY = pHandle->keyPointsY + pHandle->initParameter.maxKeyPointsNum * frameId;
        pHandle->fa->slowAlign(img, pHandle->initParameter.height, pHandle->initParameter.width, keyPointsX, keyPointsY, pHandle->initParameter.maxKeyPointsNum, pHandle->initParameter.leftEyeId, pHandle->initParameter.rightEyeId, mergeWeight);
        auto fromAlignImg = pHandle->fa->getAlignImage();
        auto cutSize = pHandle->initParameter.cutSize;
        auto alignSize = pHandle->initParameter.alignSize;
        auto toAlignImg = pHandle->alignImages + cutSize * cutSize * 3 * frameId;
        // 将原图拷贝出来
        int offset = ((alignSize - cutSize) >> 1);
        for(int lineId = 0; lineId < cutSize; ++lineId){
            memcpy(toAlignImg + lineId * cutSize * 3, fromAlignImg + ((lineId + offset) * alignSize + offset) * 3, cutSize * 3 * sizeof(unsigned char));
        }
    }
    return 0;
}

unsigned char* FA_GetAlignFace(FA_Handle* pHandle){
    return pHandle->alignImages;
}

#ifdef __cplusplus
}
#endif