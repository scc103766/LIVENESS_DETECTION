//
// Created by EDY on 2022/5/19.
//

#include "normal_cues.h"
#include <memory.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>

using namespace std;

#define GET_COLOR_1(x) ((x & 0x00FF0000) >> 16)
#define GET_COLOR_2(x) ((x & 0x0000FF00) >> 8)
#define GET_COLOR_3(x) (x & 0x000000FF)
#define GET_BRIGHT(r, g, b) (r * 0.299f + g * 0.587f + b * 0.114f)


template<class T>
T clamp(T x, T min, T max) {
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

inline void argsort3(float vs[], int ids[]){
    if(vs[0] <= vs[1] && vs[0] <= vs[2]){
        ids[0] = 0;
        if(vs[1] <= vs[2]){
            ids[1] = 1;
            ids[2] = 2;
        } else {
            ids[1] = 2;
            ids[2] = 1;
        }
    }
    else if(vs[0] > vs[1] && vs[0] > vs[2]){
        ids[0] = 2;
        if(vs[1] <= vs[2]){
            ids[1] = 0;
            ids[2] = 1;
        } else {
            ids[1] = 1;
            ids[2] = 0;
        }
    }
    else{
        ids[0] = 1;
        if(vs[1] <= vs[2]){
            ids[1] = 0;
            ids[2] = 2;
        } else {
            ids[1] = 2;
            ids[2] = 0;
        }
    }
}

inline void invArgSort3(float vs[], int ids[]){
    // 第几大的数的id是什么
    if(vs[0] <= vs[1] && vs[0] <= vs[2]){
        ids[0] = 0;
        if(vs[1] <= vs[2]){
            ids[1] = 1;
            ids[2] = 2;
        } else {
            ids[1] = 2;
            ids[2] = 1;
        }
    }else if(vs[0] > vs[1] && vs[0] > vs[2]){
        ids[2] = 0;
        if(vs[1] <= vs[2]){
            ids[0] = 1;
            ids[1] = 2;
        } else {
            ids[0] = 2;
            ids[1] = 1;
        }
    }  else {
        ids[1] = 0;
        if(vs[1] <= vs[2]){
            ids[0] = 1;
            ids[2] = 2;
        } else {
            ids[0] = 2;
            ids[2] = 1;
        }
    }
}

inline void calToK(int toK[], float channelScale[], float channelSum[]){
    // 估计自己的闪光颜色是第几大的数
    int scaleIds[3];
    argsort3(channelScale, scaleIds);
    // 估计第几大的数在几号通道
    int sumIds[3];
    invArgSort3(channelSum, sumIds);
    for(int i = 0; i < 3; ++i){
        toK[i] = sumIds[scaleIds[i]];
    }
}


NormalCues::NormalCues(int alignSize, int n, float normalizePercent)
        : alignSize(alignSize), n(n), normalizePercent(normalizePercent) {
    int oPixelCnt = alignSize * alignSize * 3;
    // 缓存像素差
    cache = new unsigned char[oPixelCnt];
}

NormalCues::~NormalCues() {
    delete[]cache;
}

inline void sumColor(unsigned char *iImage, int alignSize, int cutSize, float color[3]) {
    int minY = ((alignSize - cutSize) >> 1);
    int maxY = alignSize - minY;
    int minX = minY;
    int maxX = maxY;

    memset(color, 0, 3 * sizeof(float));

    for (int i = minY; i < maxY; ++i) {
        for (int j = minX; j < maxX; ++j) {
            auto pixel = iImage + (i * alignSize + j) * 3;
            for (int k = 0; k < 3; ++k) {
                color[k] += pixel[k];
            }
        }
    }
}

#define PINCH_DARK_BRIGHT true

unsigned char *NormalCues::calNormalCuesMap(int *screenColors, unsigned char *iImage, int cutSize, int normalizeSize, unsigned char *toImage) {
    auto bright = iImage;
    auto dark = iImage + (n - 1) * alignSize * alignSize * 3;
    if(PINCH_DARK_BRIGHT){
        // 计算最亮的像素
        float colorBright[3];
        sumColor(bright, alignSize, cutSize, colorBright);
        // 计算最暗像素
        float colorDark[3];
        sumColor(dark, alignSize, cutSize, colorDark);

        float colorMax[3] = {0, 0, 0};
        float colorMin[3] = {1e32, 1e32, 1e32};
        float colorTmp[3];
        // 计算中间像素的最亮值和最暗值
        for (int i = 1; i < n - 1; ++i) {
            sumColor(iImage + i * alignSize * alignSize * 3, alignSize, cutSize, colorTmp);
            for (int k = 0; k < 3; ++k) {
                colorMax[k] = max(colorTmp[k], colorMax[k]);
                colorMin[k] = min(colorTmp[k], colorMin[k]);
            }
        }

        // 调整最亮和最暗像素
        int minY = ((alignSize - cutSize) >> 1);
        int maxY = alignSize - minY;
        int minX = minY;
        int maxX = maxY;
        for (int i = minY; i < maxY; ++i) {
            for (int j = minX; j < maxX; ++j) {
                auto brightPixel = bright + (i * alignSize + j) * 3;
                auto darkPixel = dark + (i * alignSize + j) * 3;
                for (int k = 0; k < 3; ++k) {
                    brightPixel[k] = (unsigned char) (brightPixel[k] * min(colorMax[k] / colorBright[k], 1.f));
                    darkPixel[k] = (unsigned char) (clamp<float>(darkPixel[k] * max(colorMin[k] / colorDark[k], 1.f), 0.f,
                                                                 255.f));
                }
            }
        }
    }


    for (int i = 1; i < n - 1; ++i) {
        auto curImage = iImage + i * alignSize * alignSize * 3;
        auto tImg = toImage + (i - 1) * cutSize * cutSize * 3 * 2;
        calNormalCues(bright, dark, curImage, tImg, screenColors[i], cutSize, normalizeSize);
    }
    // 返回最多非法像素占比
    return toImage;
}

//#define EXCLUDE_HALF_DARK_DELTA

#ifdef EXCLUDE_HALF_DARK_DELTA

inline void fillChannel(unsigned char* bright, unsigned char* dark, unsigned char* toImage, unsigned char* cache, int screenColor, int dScreenColor, int alignSize, int cutSize, int normalizeSize, float normalizePercent){
    int fromColor[3];
    // bgr
    fromColor[0] = GET_COLOR_1(screenColor);
    fromColor[1] = GET_COLOR_2(screenColor);
    fromColor[2] = GET_COLOR_3(screenColor);
    int deltaColor[3];
    deltaColor[0] = GET_COLOR_1(dScreenColor);
    deltaColor[1] = GET_COLOR_2(dScreenColor);
    deltaColor[2] = GET_COLOR_3(dScreenColor);


    int pixelNum = alignSize * alignSize;

    int minY = ((alignSize - cutSize) >> 1);
    int maxY = alignSize - minY;
    int minX = minY;
    int maxX = maxY;

    int norMinY = ((alignSize - normalizeSize) >> 1);
    int norMaxY = alignSize - norMinY;
    int norMinX = norMinY;
    int norMaxX = norMaxY;

    // 归一化的参数
    unsigned char* subCache[] = {cache, cache + pixelNum, cache + 2 * pixelNum};
    int subCacheSize[] = {0, 0, 0};

    // 开始填写
    memset(cache, 0, pixelNum * 3 * sizeof(unsigned char ));
    // 填写通道，以便排序归一化
    for(int i = norMinY; i < norMaxY; ++i){
        for(int j = norMinX; j < norMaxX; ++j){
            int pOffset = (i * alignSize + j) * 3;
            auto pBright = bright + pOffset;
            auto pDark = dark + pOffset;
            for(int k = 0; k < 3; ++k){
                if(pBright[k] > pDark[k]){
                    //cache[cacheSize++] = pFrom[k] - pDark[k];
                    unsigned char t = pBright[k] - pDark[k];
                    subCache[k][subCacheSize[k]++] = t;
                } /*else if(deltaColor[k] < 30) {
                    subCache[k][subCacheSize[k]++] = 0;
                }*/
            }
        }
    }

    // 全集的归一化系数，认为暗光如果产生较多的像素差，这些像素就是伪的
    unsigned char minC = 255;
    unsigned char maxC = 0;
    // 局部的归一化系数
    unsigned char subMinC[] = {0, 0, 0};
    unsigned char subMaxC[] = {0, 0, 0};
    // 得到归一化参数和归一化后的像素和
    for(int k = 0; k < 3; ++k){
        sort(subCache[k], subCache[k] + subCacheSize[k]);
        if(deltaColor[k] < 30){
            // 如果是暗光，暗光不可能产生太多色差，直接排除掉0.32%的光差像素
            minC = min(subCache[k][int(subCacheSize[k] * 0.32f)], minC);
        } else if(deltaColor[k] > 250) {
            // 如果是中光或者强光
            maxC = max(subCache[k][int(subCacheSize[k] * normalizePercent)], maxC);
        }
        subMinC[k] = subCache[k][int(subCacheSize[k] * (1 - normalizePercent))];
        subMaxC[k] = subCache[k][int(subCacheSize[k] * (normalizePercent))];
    }

    // 统计归一化后的像素累加和
    float subChannelSum[] = {0, 0, 0};

    for(int k = 0; k < 3; ++k){
        if(subMinC[k] < minC)
            subMinC[k] = minC;
        if(subMaxC[k] > maxC)
            subMinC[k] = maxC;
        float delta = subMaxC[k] - subMinC[k] + 1e-6;
        for(int i = 0; i < subCacheSize[k]; ++i){
            subChannelSum[k] += clamp<float>((subCache[k][i] - (float)subMinC[k]) / delta, 0.f, 1.f);
        }
    }

    // 通道切换，将像素和最大的通道放到闪光最强的颜色对应通道上
    int toK[] = {0, 0, 0};
    float s = logf(fromColor[0]) + logf(fromColor[1]) + logf(fromColor[2]);

    // 期望的闪光强度比
    float channelScale[] = {logf(fromColor[0]) / s, logf(fromColor[1]) / s, logf(fromColor[2]) / s};
    // k=tok[i] 表示期望将第k个通道的像素放到第i个通道上，作为输出
    calToK(toK, channelScale, subChannelSum);

    // 原始的色差强度比
    s = subChannelSum[0] + subChannelSum[1] + subChannelSum[2];
    subChannelSum[0] /= s;
    subChannelSum[1] /= s;
    subChannelSum[2] /= s;

    // k=toK[i]，将第k个通道输出到第i个后，为了将像素和占比调整成channelScale[i]，需要缩放多少倍
    float maxChannelScale = 0;
    for (int i = 0; i < 3; ++i) {
        channelScale[i] = channelScale[i] / subChannelSum[toK[i]];
        maxChannelScale = max(maxChannelScale, channelScale[i]);
    }

    // 将最大缩放系数变成1
    for (int i = 0; i < 3; ++i)
        channelScale[i] /= maxChannelScale;


    int outId = 0;
    for (int i = minY; i < maxY; ++i) {
        for (int j = minX; j < maxX; ++j) {
            int pOffset = (i * alignSize + j) * 3;
            auto pBright = bright + pOffset;
            auto pDark = dark + pOffset;
            auto pTo = toImage + outId * 3;
            ++outId;
            for (int sk = 0; sk < 3; ++sk) {
                int k = toK[sk];
                if(pBright[k] > pDark[k]){
                    pTo[sk] = (unsigned char) (clamp<unsigned char>(pBright[k] - pDark[k], subMinC[k], subMaxC[k]) - subMinC[k]) * 255.f * channelScale[sk] / (subMaxC[k] - subMinC[k] + 1e-6);
                } else {
                    pTo[sk] = 0;
                }
            }
        }
    }
}

#else

inline void fillChannel(unsigned char* bright, unsigned char* dark, unsigned char* toImage, unsigned char* cache, int screenColor, int dScreenColor, int alignSize, int cutSize, int normalizeSize, float normalizePercent){
    int fromColor[3];
    // bgr
    fromColor[0] = GET_COLOR_1(screenColor);
    fromColor[1] = GET_COLOR_2(screenColor);
    fromColor[2] = GET_COLOR_3(screenColor);
    int deltaColor[3];
    deltaColor[0] = GET_COLOR_1(dScreenColor);
    deltaColor[1] = GET_COLOR_2(dScreenColor);
    deltaColor[2] = GET_COLOR_3(dScreenColor);

    int pixelNum = alignSize * alignSize;

    int minY = ((alignSize - cutSize) >> 1);
    int maxY = alignSize - minY;
    int minX = minY;
    int maxX = maxY;

    int norMinY = ((alignSize - normalizeSize) >> 1);
    int norMaxY = alignSize - norMinY;
    int norMinX = norMinY;
    int norMaxX = norMaxY;

    // 归一化的参数
    unsigned char* subCache[] = {cache, cache + pixelNum, cache + 2 * pixelNum};
    int subCacheSize[] = {0, 0, 0};
    unsigned char subMinC[] = {0, 0, 0};
    unsigned char subMaxC[] = {0, 0, 0};
    float subChannelSum[] = {0, 0, 0};
    float channelScale[] = {logf(fromColor[0]), logf(fromColor[1]), logf(fromColor[2])};

    // 开始填写
    memset(cache, 0, pixelNum * 3 * sizeof(unsigned char ));
    // 填写通道，以便排序归一化
    for(int i = norMinY; i < norMaxY; ++i){
        for(int j = norMinX; j < norMaxX; ++j){
            int pOffset = (i * alignSize + j) * 3;
            auto pBright = bright + pOffset;
            auto pDark = dark + pOffset;
            for(int k = 0; k < 3; ++k){
                if(pBright[k] > pDark[k]){
                    unsigned char t = pBright[k] - pDark[k];
                    subCache[k][subCacheSize[k]++] = t;
                }/*else if(deltaColor[k] < 30) {
                    subCache[k][subCacheSize[k]++] = 0;
                }*/
            }
        }
    }
    // 得到归一化参数和归一化后的像素和
    for(int k = 0; k < 3; ++k){
        sort(subCache[k], subCache[k] + subCacheSize[k]);

        /*if(deltaColor[k] < 30) {
            subMinC[k] = subCache[k][int(subCacheSize[k] * 0.5f)];
        } else {
            subMinC[k] = subCache[k][int(subCacheSize[k] * (1 - normalizePercent))];
        }*/
        subMinC[k] = subCache[k][int(subCacheSize[k] * (1 - normalizePercent))];


        subMaxC[k] = subCache[k][int(subCacheSize[k] * (normalizePercent))];
        float delta = subMaxC[k] - subMinC[k];
        for(int i = 0; i < subCacheSize[k]; ++i){
            subChannelSum[k] += clamp<float>((subCache[k][i] - (float)subMinC[k]) / delta, 0.f, 1.f);
        }
    }

    // 通道切换，将像素和最大的通道放到闪光最强的颜色对应通道上
    int toK[] = {0, 0, 0};
    calToK(toK, channelScale, subChannelSum);

    float allSum = subChannelSum[0] + subChannelSum[1] + subChannelSum[2];
    float s = channelScale[0] + channelScale[1] + channelScale[2];
    float maxChannelScale = 0;
    for (int i = 0; i < 3; ++i) {
        channelScale[i] = channelScale[i] / s * allSum / subChannelSum[toK[i]];
        maxChannelScale = max(maxChannelScale, channelScale[i]);
    }

    for (int i = 0; i < 3; ++i)
        channelScale[i] /= maxChannelScale;

    int outId = 0;
    for (int i = minY; i < maxY; ++i) {
        for (int j = minX; j < maxX; ++j) {
            int pOffset = (i * alignSize + j) * 3;
            auto pBright = bright + pOffset;
            auto pDark = dark + pOffset;
            auto pTo = toImage + outId * 3;
            ++outId;
            for (int sk = 0; sk < 3; ++sk) {
                int k = toK[sk];
                if(pBright[k] > pDark[k]){
                    pTo[sk] = (unsigned char) (clamp<unsigned char>(pBright[k] - pDark[k], subMinC[k], subMaxC[k]) - subMinC[k]) * 255.f * channelScale[sk] / (subMaxC[k] - subMinC[k] + 1e-6);
                } else {
                    pTo[sk] = 0;
                }
            }
        }
    }
}

#endif


void NormalCues::calNormalCues(unsigned char *brightImage, unsigned char *darkImage, unsigned char *iImage,
                               unsigned char *toImage, int screenColor, int cutSize, int normalizeSize) {

    // 先填写pFrom-pDark------------------------------------------------------------
    fillChannel(iImage, darkImage, toImage, cache, screenColor, screenColor, alignSize, cutSize, normalizeSize, normalizePercent);
    // 再填写pBright-pFrom------------------------------------------------------------
    int dScreenColor = ((255 - GET_COLOR_1(screenColor)) << 16) + ((255 - GET_COLOR_2(screenColor)) << 8) + (255 -
            GET_COLOR_3(screenColor));
    fillChannel(brightImage, iImage, toImage + cutSize * cutSize * 3, cache, screenColor, dScreenColor, alignSize, cutSize, normalizeSize, normalizePercent);
}