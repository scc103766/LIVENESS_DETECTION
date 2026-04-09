//
// Created by yanyu on 2022/5/20.
//

#include "libtg_api.h"

extern "C"
{
    // 创建程序对象
    DLLEXPORT TG* createLibtg(int height, int width, unsigned char *mask, int alignSize, int n, float normalizePercent){
        return new TG(height, width, mask, alignSize,  n, normalizePercent);
    }

    // 释放程序对象
    DLLEXPORT void releaseLibtg(TG* libtg){
        delete libtg;
    }

    // 是否选择当前图片
    DLLEXPORT bool isInsertImg(TG* libtg, unsigned char *img, int screenColor, int state, int step){
        return libtg->isInsertImg(img, screenColor, state, step);
    }

    // 计算法线信息图
    DLLEXPORT unsigned char *calNormalCuesMap(TG* libtg, int* screenColors, unsigned char* iImage, int cutSize, int normalizeSize){
        return libtg->calNormalCuesMap(screenColors, iImage, cutSize, normalizeSize);
    }

}