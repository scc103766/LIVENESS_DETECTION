//
// Created by EDY on 2022/5/23.
//
#include "nx_tg_export.h"
#include "tg.h"
#include "memory.h"
#include <mutex>
//#include <iostream>
//using namespace std;

#define NORMAL_CUE_SIZE 256

struct TG_Handle {
    TG *tg;

    int *screenColor;

    int curInsertImageId;

    int pixelCnt;

    int flashNum;

    // 归一化像素比例，默认设置0.9
    float normalizePercent;

    // 保护创建和释放资源
    std::mutex mtx;
};

#ifdef __cplusplus
extern "C" {
#endif

int TG_Initial(TG_Handle **ppHandle, TG_Init_Parameter initParameter) {
    if (initParameter.flashNum != 5) {
        // ERROR：目前只支持5次闪光
        return 1;
    }
    if (initParameter.height < initParameter.alignSize || initParameter.width < initParameter.alignSize) {
        // ERROR：图片尺寸太小
        return 2;
    }
    (*ppHandle) = new TG_Handle();
    auto pHandle = (*ppHandle);
    pHandle->tg = new TG(initParameter.height, initParameter.width, initParameter.mask, initParameter.alignSize, initParameter.flashNum, initParameter.normalizePercent);

    pHandle->pixelCnt = initParameter.height * initParameter.width;
    pHandle->flashNum = initParameter.flashNum;

    pHandle->screenColor = new int[initParameter.flashNum];

    return 0;
}

const char *TG_GetVersion() { return "1.0.0.220727"; }

int TG_Uninitial(TG_Handle *pHandle) {
    delete[] pHandle->screenColor;
    delete pHandle->tg;
    return 0;
}

int TG_IsInsertImage(TG_Handle *pHandle, TG_Insert_Parameter insertParameter) {
    if(insertParameter.encode < 0 || insertParameter.encode > 1)
        return 4;
    pHandle->mtx.lock();
    // 推理当前插入的图片id
    if (insertParameter.state == 0) {
        // 当前插入的是白屏的第一帧或者黑屏的第一帧
        if (insertParameter.screenColor != 0) {
            // 当前插入的是白屏的第一帧
            pHandle->curInsertImageId = 0;
        } else {
            // 当前插入的是黑屏的第一帧
            ++pHandle->curInsertImageId;
        }
    } else if (insertParameter.state == 1) {
        ++pHandle->curInsertImageId;
    } else {
        // 插入非第一帧图片
        if (insertParameter.screenColor != pHandle->screenColor[pHandle->curInsertImageId]) {
            pHandle->mtx.unlock();
            // ERROR：插入的颜色不对
            return 2;
        }
    }

    if (pHandle->curInsertImageId >= pHandle->flashNum) {
        // ERROR：插入太多图片
        pHandle->mtx.unlock();
        return 1;
    }


    // 记录从视频中选择的合适的图片
    if (pHandle->tg->isInsertImg(insertParameter.img, insertParameter.screenColor, insertParameter.state,
                                 insertParameter.step, insertParameter.encode, insertParameter.rotate)) {
        pHandle->screenColor[pHandle->curInsertImageId] = insertParameter.screenColor;
        pHandle->mtx.unlock();
        return 0;
    }

    pHandle->mtx.unlock();
    return 3;
}


int TG_CalNormalCues(TG_Handle *pHandle, unsigned char* alignImages) {
    pHandle->mtx.lock();
    if (pHandle->curInsertImageId != pHandle->flashNum - 1) {
        // ERROR：插入的图片不够
        pHandle->curInsertImageId = 0;
        pHandle->mtx.unlock();
        return 1;
    }
    pHandle->curInsertImageId = 0;

    // 计算法线信息图
    pHandle->tg->calNormalCuesMap(pHandle->screenColor, alignImages, NORMAL_CUE_SIZE, NORMAL_CUE_SIZE);

    pHandle->mtx.unlock();
    return 0;
}

unsigned char *TG_GetNormalCues(TG_Handle *pHandle) {
    return pHandle->tg->getNormalCuesMap();
}

#ifdef __cplusplus
}
#endif