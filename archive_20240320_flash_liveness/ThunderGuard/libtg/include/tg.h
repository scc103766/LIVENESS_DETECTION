//
// Created by yanyu on 2022/5/20.
// 适配器模式，负责调用底下所有功能
//

#ifndef LIBTG_TG_H
#define LIBTG_TG_H

#include "img_choose.h"
#include "normal_cues.h"

#define MAX_N 256

class TG {
public:
    TG(int height, int width, unsigned char *mask, int alignSize=480, int n=5, float normalizePercent=0.95f);
    virtual ~TG();
    bool isInsertImg(unsigned char *img, int screenColor, int state, int step=2, int encode=0, int rotate=0);
    // 计算法线信息图，将法线信息图裁剪成cutSize*cutSize，统计中心的normalizeSize*normalizeSize，做像素归一化！cutSize>=normalizeSize
    unsigned char* calNormalCuesMap(int* screenColors, unsigned char* iImage, int cutSize, int normalizeSize);
    unsigned char* getNormalCuesMap();
private:
    int alignSize;
    int height;
    int width;
    int n;
protected:
    // 负责从视频中挑选n张最合适的图
    ImgChoose imgChoose;
    NormalCues normalCues;
    // 缓存对齐图或者法线信息图
    unsigned char *cache;
};


#endif //LIBTG_TG_H
