#include "libfa_api.h"

// 创建程序对象
DLLEXPORT FA* createLibFA(int alignSize, int faceSize, int maxAlignOffsetPixel, int maxKeyPointsNum, float normalizePercent){
    return new FA(alignSize, faceSize, maxAlignOffsetPixel, maxKeyPointsNum, normalizePercent);
}

// 释放程序对象
DLLEXPORT void releaseLibFA(FA* libfa){
    delete libfa;
}


DLLEXPORT float* getTransParams(FA* libfa){return libfa->getTransParams();}
DLLEXPORT unsigned char* getAlignImage(FA* libfa){return libfa->getAlignImage();}
DLLEXPORT unsigned char* getDepthMap(FA* libfa){return libfa->getDepthMap();}

DLLEXPORT void cutFace(FA* libfa, unsigned char* img, int height, int width, float* transParams){
    libfa->cutFace(img, height, width, transParams);
}

DLLEXPORT void fastAlign(FA* libfa, unsigned char* img, int height, int width, int leftEyeX, int leftEyeY, int rightEyeX, int rightEyeY){
    libfa->fastAlign(img, height, width, leftEyeX, leftEyeY, rightEyeX, rightEyeY);
}
DLLEXPORT void slowAlign(FA* libfa, unsigned char* img, int height, int width, int* keyPointsX, int* keyPointsY, int keyPointsNum, int leftEyeId, int rightEyeId, float mergeWeight){
    libfa->slowAlign(img, height, width, keyPointsX, keyPointsY, keyPointsNum, leftEyeId, rightEyeId, mergeWeight);
}
DLLEXPORT void pixelInterpolation(FA* libfa, float* vertices, int verticesNum){
    libfa->pixelInterpolation(vertices, verticesNum);
}