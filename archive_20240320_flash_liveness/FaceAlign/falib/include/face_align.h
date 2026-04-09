//
// Created by Yanyu on 2022/7/14.
//

#ifndef FALIB_FACE_ALIGN_H
#define FALIB_FACE_ALIGN_H

class FaceAlign{
public:
    // 初始化，对齐后的图片大小是alignSize，人脸区域的大小是faceSize，最大帧数是maxFrameNum
    // 对齐时，左眼位于人脸区域的左1/4位，右眼位于人脸区域的右1/4位，两眼都位于人脸区域的上1/4位
    // 在原始图随机抖动做像素偏移，然后实现像素级别的对齐。最大偏移数量maxAlignOffsetPixel
    FaceAlign(int alignSize=384, int faceSize=288, int maxAlignOffsetPixel=4, int maxKeyPointsNum=256);
    virtual ~FaceAlign();
    // 仅仅根据眼睛位置快速人脸对齐，返回对齐后的转移参数和图片
    void fastAlign(unsigned char* img, int height, int width, int leftEyeX, int leftEyeY, int rightEyeX, int rightEyeY, float* outTransParams, unsigned char* outAlignImg= nullptr);

    // 进行慢速对齐，同样返回对齐后的转移参数和图片
    // 使用mergeWeight来控制对齐融合，如果mergeWeight<0，表示不使用融合，直接覆盖锚图；mergeWeight>=0，在对齐完成后，锚图=锚图*(1-mergeWeight)+对齐图*mergeWeight
    void slowAlign(unsigned char* img, int height, int width, int* keyPointsX, int* keyPointsY, int keyPointsNum, int leftEyeId, int rightEyeId, float mergeWeight, float* outTransParams, unsigned char* outAlignImg= nullptr);

    // 根据对齐产生，裁剪图片
    void cutFace(unsigned char* img, int height, int width, float* transParams, unsigned char* outAlignImg);

    int getAlignSize(){return alignSize;}

protected:
    void copyToAnchorImage(unsigned char* alignImage, unsigned char* toAnchorColorCache, unsigned char* toAnchorSobelCache);
    float calKeyPointLoss(int keyPointNum);
    float calColorLoss();
    void mergeAnchor(float mergeWeight);
protected:
    int alignSize;
    int faceSize;
    int maxAlignOffsetPixel;
    int maxKeyPointsNum;

    // 缓存对齐后的图
    unsigned char* alignImageCache;

    int* anchorKeyPointsXCache;
    int* anchorKeyPointsYCache;

    // 用来通过颜色对比，来评估对齐效果
    unsigned char* anchorColorCache;

    // 用来通过边缘对比，来评估对齐效果
    unsigned char* anchorSobelCache;
    float *sobelCache;

    //标记已经计算过的偏移
    int *flag;
};

#endif //FALIB_FACE_ALIGN_H
