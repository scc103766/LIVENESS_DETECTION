//
// Created by Yanyu on 2022/7/14.
//
#include "img_proc.h"
#include "string.h"
#include <cmath>
#include <algorithm>

using namespace std;


void ImgProc::bilinera(unsigned char *fromImage, int width, float fi, float fj, unsigned char *toPixel, int channel) {
    // 双线性插值，参考代码：https://blog.csdn.net/weixin_45116749/article/details/119644866
    // 在x轴方向向上向下取整
    int j1 = floor(fj);
    int j2 = ceil(fj);
    float u = fj - j1;
    // y方向重新做一次
    int i1 = floor(fi);
    int i2 = ceil(fi);
    float v = fi - i1;
    // 得到四周围像素
    auto p1 = fromImage + (i1 * width + j1) * channel;
    auto p2 = fromImage + (i2 * width + j1) * channel;
    auto p3 = fromImage + (i1 * width + j2) * channel;
    auto p4 = fromImage + (i2 * width + j2) * channel;
    auto s1 = (1-u)*(1-v);
    auto s2 = (1-u)*v;
    auto s3 = u * (1-v);
    auto s4 = u * v;
    for(int c = 0; c < channel; ++c){
        float tmp = s1 * p1[c] + s2 * p2[c] + s3 * p3[c] + s4 * p4[c];
        toPixel[c] = (unsigned char)(tmp < 0.f ? 0.0f : (tmp > 255.f ? 255.f: tmp));
    }
}

void ImgProc::sobel(unsigned char *iImage, int height, int width, float* toSobelCache){
    memset(toSobelCache, 0, height * width * 3 * sizeof(float ));
    float color[3] = {0, 0, 0};
    for(int i = 1; i < height-1; ++i){
        int hOffset = width * 3;
        auto u = iImage + (i-1) * hOffset;
        auto m = iImage + i * hOffset;
        auto d = iImage + (i+1) * hOffset;
        for(int j = 1; j < width-1; ++j){
            for(int k = 0; k < 3; ++k){
                float gx = abs((u[(j-1) * 3 + k] * 1.f + m[(j-1) * 3 + k] * 2.f + d[(j-1) * 3 + k] * 1.f) - (u[(j+1) * 3 + k] * 1.0 + m[(j+1) * 3 + k] * 2.f + d[(j+1) * 3 + k] * 1.f));
                float gy = abs((u[(j-1) * 3 + k] * 1.f + u[j * 3 + k] * 2.f + u[(j+1) * 3 + k] * 1.f) - (d[(j-1) * 3 + k] * 1.f + d[j * 3 + k] * 2.f + d[(j+1) * 3 + k] * 1.f));
                toSobelCache[(i * height + j) * 3 + k] = gx + gy;
                color[k] = fmax(color[k], gx + gy);
            }
        }
    }
    for(int i = 0; i < height * width; ++i){
        for(int k = 0; k < 3; ++k){
            toSobelCache[i * 3 + k] = toSobelCache[i * 3 + k] / color[k];
        }
    }
}

void ImgProc::float2uchar(float *iImage, int height, int width, unsigned char *toImage) {
    for(int i = 0; i < height * width; ++i){
        for(int k = 0; k < 3; ++k){
            toImage[i * 3 + k] = (unsigned char)(iImage[i * 3 + k] * 255);
        }
    }
}

void ImgProc::normalize(unsigned char *image, unsigned char *cache, int height, int width, int channel, float percent) {
    for(int c = 0; c < channel; ++c){
        int pixelId = 0;
        for(int i = 0; i < height; ++i){
            int offset = i * width * channel;
            for(int j = 0; j < width; ++j){
                cache[pixelId++] = image[offset + j * channel + c];
            }
        }
        sort(cache, cache + pixelId);
        int minId = int(pixelId * (1 - percent));
        if(minId < 0)
            minId = 0;
        int maxId = int(pixelId * percent);
        if(maxId >= pixelId)
            maxId = pixelId - 1;
        unsigned char minI = cache[minId];
        unsigned char maxI = cache[maxId];

        for(int i = 0; i < pixelId; ++i){
            int curId = pixelId * channel + c;
            if(image[curId] <= minI)
                image[curId] = 0;
            else if(image[curId] >= maxI)
                image[curId] = 255;
            else
                image[curId] = (unsigned char)(float(image[curId] - minI) * 255 / float(maxI - minI));
        }
    }
}
