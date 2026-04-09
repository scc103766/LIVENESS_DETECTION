//
// Created by yanyu on 2022/5/20.
//

#include "tg.h"
#include <string.h>
#include <iostream>
using namespace std;


TG::TG(int height, int width, unsigned char *mask, int alignSize, int n, float normalizePercent)
        : alignSize(alignSize), height(height), width(width), n(n), imgChoose(height, width, mask),
           normalCues(alignSize, n, normalizePercent) {
    cache = new unsigned char[alignSize * alignSize * 3 * (n - 2) * 2];
}

TG::~TG() { delete[]cache; }

bool TG::isInsertImg(unsigned char *img, int screenColor, int state, int step, int encode, int rotate) {
    return imgChoose.isInsertImg(img, screenColor, state, step, encode, rotate);
}

unsigned char *TG::calNormalCuesMap(int* screenColors, unsigned char *iImage, int cutSize, int normalizeSize) {
    return normalCues.calNormalCuesMap(screenColors, iImage, cutSize, normalizeSize, cache);
}

unsigned char* TG::getNormalCuesMap() {
    return cache;
}