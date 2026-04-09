//
// Created by Yanyu on 2022/7/14.
//

#ifndef FALIB_IMG_PROC_H
#define FALIB_IMG_PROC_H

class ImgProc{
public:
    static void bilinera(unsigned char* fromImage, int width, float fi, float fj, unsigned char* toPixel, int channel);
    static void sobel(unsigned char *iImage, int height, int width, float* toSobelCache);
    static void float2uchar(float* iImage, int height, int width, unsigned char* toImage);
    static void normalize(unsigned char* image, unsigned char* cache, int height, int width, int channel, float percent);
};

#endif //FALIB_IMG_PROC_H
