//
// Created by yanyu on 2022/5/18.
//

#include "img_choose.h"
#include <math.h>

#define GET_COLOR_1(x) ((x & 0x00FF0000) >> 16)
#define GET_COLOR_2(x) ((x & 0x0000FF00) >> 8)
#define GET_COLOR_3(x) (x & 0x000000FF)
#define GET_BRIGHT(r, g, b) (r * 0.299f + g * 0.587f + b * 0.114f)
#define MIN(x,y)(x<y)?(x):(y)
#define MAX(x,y)(x>y)?(x):(y)


inline static void trans(int rotate, int height, int width, int i, int j, int& toI, int& toJ){
    switch (rotate) {
        case 90:
            toI = j;
            toJ = height - i - 1;
            break;
        case 180:
            toI = height - i - 1;
            toJ = width - j - 1;
            break;
        case 270:
            toI = width - j - 1;
            toJ = i;
            break;
        default:
            toI = i;
            toJ = j;
    }
}

inline static void sumColorYUV420SP(unsigned char* yuv420sp, unsigned char *mask, int width, int height, int rotate, int minX, int maxX, int minY, int maxY, int step, float &sumB, float &sumG, float &sumR)
{
    int frameSize = width * height;
    int toHeight = (rotate == 90 || rotate == 270) ? width : height;
    int toWidth = (rotate == 90 || rotate == 270) ? height : width;

    int i = 0, j = 0, y = 0;
    int toI = 0, toJ = 0;
    int yp = 0, uvp = 0, u = 0, v = 0;
    int y1192 = 0, r = 0, g = 0, b = 0;
    sumR = 0;
    sumG = 0;
    sumB = 0;

    for (int i = minY; i < maxY; i+=step){
        for (j = minX; j < maxX; j+=step){
            int id = i * width + j;
            if(mask[id] > 0){
                trans(rotate, height, width, i, j, toI, toJ);
                yp = toI * toWidth + toJ;
                uvp = frameSize + (toI >> 1) * toWidth + (toJ & ~1);

                y = (0xff & ((int) yuv420sp[yp])) - 16;
                if (y < 0) y = 0;
                v = (0xff & yuv420sp[uvp]) - 128;
                u = (0xff & yuv420sp[uvp+1]) - 128;
                y1192 = 1192 * y;
                r = (y1192 + 1634 * v);
                g = (y1192 - 833 * v - 400 * u);
                b = (y1192 + 2066 * u);

                if (r < 0) r = 0; else if (r > 262143) r = 262143;
                if (g < 0) g = 0; else if (g > 262143) g = 262143;
                if (b < 0) b = 0; else if (b > 262143) b = 262143;

                sumR += (float )(r >> 10);
                sumG += (float )(g >> 10);
                sumB += (float )(b >> 10);
            }
        }
    }
}

inline static void sumColorBGR(unsigned char* img, unsigned char *mask, int width, int height, int minX, int maxX, int minY, int maxY, int step, float &sumB, float &sumG, float &sumR){
    sumR = 0;
    sumG = 0;
    sumB = 0;
    for(int i = minY; i < maxY; i += step){
        for(int j = minX; j < maxX; j += step){
            int id = i * width + j;
            if(mask[id] > 0){
                // bgr
                auto curPixel = img + id * 3;
                sumB += curPixel[0];
                sumG += curPixel[1];
                sumR += curPixel[2];
            }
        }
    }
}

ImgChoose::ImgChoose(int height, int width, unsigned char *mask) : height(height), width(width) {
    this->mask = new unsigned char [height * width];
    // 填写mask，并记录minX/Y和maxX/Y
    minY = height;
    minX = width;
    maxY = 0;
    maxX = 0;
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            int id = i * width + j;
            this->mask[id] = mask[id * 3];
            if(this->mask[id] > 0){
                minY = MIN(minY, i);
                maxY = MAX(maxY, i);
                minX = MIN(minX, j);
                maxX = MAX(maxX, j);
            }
        }
    }
}

ImgChoose::~ImgChoose() {
    delete []mask;
}

bool ImgChoose::isInsertImg(unsigned char *img, int screenColor, int state, int step, int encode, int rotate) {
    float curFlag = 0;

    float colorSum[] = {0.f, 0.f, 0.f};
    if(encode == 0){
        sumColorBGR(img, mask, width, height, minX, maxX, minY, maxY, step, colorSum[0], colorSum[1], colorSum[2]);
    } else if(encode == 1){
        sumColorYUV420SP(img, mask, width, height, rotate, minX, maxX, minY, maxY, step, colorSum[0], colorSum[1], colorSum[2]);
    } else {
        return false;
    }


    if(state == 1 || state == 4){
        float colorUpScale[] = {0.f, 0.f};
        float colorDownScale[] = {0.f,  0.f};
        unsigned char color[] = {0, 0, 0};
        color[0] = GET_COLOR_1(screenColor);
        color[1] = GET_COLOR_2(screenColor);
        color[2] = GET_COLOR_3(screenColor);
        for(int k = 0; k < 3; ++k){
            if(color[k] == 255){
                colorUpScale[0] = colorSum[k];
                colorUpScale[1] += 1.f;
            } else if(color[k] == 20) {
                colorDownScale[0] = colorSum[k];
                colorDownScale[1] += 1.f;
            }
        }
        curFlag = (colorUpScale[0] / colorUpScale[1]) / (colorDownScale[0] / colorDownScale[1]);
        if(colorUpScale[1] < 1e-6 || colorDownScale[1] < 1e-6)
            return false;
    } else {
        // 得到亮度
        curFlag = GET_BRIGHT(colorSum[2], colorSum[1], colorSum[0]);
    }

    if(state == 0 || state == 1){
        flag = curFlag;
        return true;
    } else if(state == 3){
        if(curFlag < flag){
            flag = curFlag;
            return true;
        }
    } else {
        if (curFlag > flag){
            flag = curFlag;
            return true;
        }
    }
    return false;
}
