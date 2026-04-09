//
// Created by yanyu on 2022/5/22.
//

#ifndef TG_API_LIBTG_API_H
#define TG_API_LIBTG_API_H

#include "tg.h"

#ifndef WIN32 // or something like that...
#define DLLEXPORT
#else
#define DLLEXPORT __declspec(dllexport)
#endif

extern "C"
{
// 创建程序对象，height=1920,width=1080，mask用来快速选图
DLLEXPORT TG* createLibtg(int height, int width, unsigned char *mask, int alignSize=480, int n=5, float normalizePercent=0.95f);

// 释放程序对象
DLLEXPORT void releaseLibtg(TG* libtg);

// 是否选择当前图片
DLLEXPORT bool isInsertImg(TG* libtg, unsigned char *img, int screenColor, int state, int step);

// 计算法线信息图
DLLEXPORT unsigned char* calNormalCuesMap(TG* libtg, int* screenColors, unsigned char* iImage, int cutSize, int normalizeSize);

}

#endif //TG_API_LIBTG_API_H
