#ifndef FALIB_LIBFA_API_H
#define FALIB_LIBFA_API_H

#include "fa.h"

#ifndef WIN32 // or something like that...
#define DLLEXPORT
#else
#define DLLEXPORT __declspec(dllexport)
#endif

extern "C"
{
// 创建程序对象
DLLEXPORT FA* createLibFA(int alignSize=384, int faceSize=288, int maxAlignOffsetPixel=4, int maxKeyPointsNum=256, float normalizePercent=0.95f);

// 释放程序对象
DLLEXPORT void releaseLibFA(FA* libfa);


DLLEXPORT float* getTransParams(FA* libfa);
DLLEXPORT unsigned char* getAlignImage(FA* libfa);
DLLEXPORT unsigned char* getDepthMap(FA* libfa);

DLLEXPORT void cutFace(FA* libfa, unsigned char* img, int height, int width, float* transParams);

DLLEXPORT void fastAlign(FA* libfa, unsigned char* img, int height, int width, int leftEyeX, int leftEyeY, int rightEyeX, int rightEyeY);
DLLEXPORT void slowAlign(FA* libfa, unsigned char* img, int height, int width, int* keyPointsX, int* keyPointsY, int keyPointsNum, int leftEyeId, int rightEyeId, float mergeWeight);
DLLEXPORT void pixelInterpolation(FA* libfa, float* vertices, int verticesNum);
}

#endif //FALIB_LIBFA_API_H
