//
// Created by yanyu on 2022/5/18.
// 从视频中挑选出最合适的几张照片
//

#ifndef LIBAG_IMG_CHOOSE_H
#define LIBAG_IMG_CHOOSE_H


class ImgChoose {
public:
    ImgChoose(int height, int width, unsigned char *mask);
    virtual ~ImgChoose();
    /*
     * 判断是否选择当前图片
     * state:
     *  0表示直接插入，并记录亮度
     *  1表示直接插入，并记录颜色差异
     *  2表示如果更亮就插入，并记录亮度
     *  3表示如果更暗就插入，并记录亮度
     *  4表示如果颜色差异更大就插入，并记录颜色差异
     * step:
     *  判断像素是否合适的步长，步长越大，效率越高，精度越低
     * */
    bool isInsertImg(unsigned char *img, int screenColor, int state, int step=2, int encode=0, int rotate=0);
protected:
    int height;
    int width;
    // 标记插入图像的信息，比如亮度
    float flag;
    int minX, maxX, minY, maxY;
    unsigned char *mask;

};


#endif //LIBAG_IMG_CHOOSE_H
