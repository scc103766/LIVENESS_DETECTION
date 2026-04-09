//
// Created by Yanyu on 2022/7/14.
//

#ifndef FALIB_FACE_DEPTH_MAP_H
#define FALIB_FACE_DEPTH_MAP_H

class FaceDepthMap{
public:
    // 填入需要计算深度图的原始图宽高
    FaceDepthMap(int height, int width);
    virtual ~FaceDepthMap();
    // 插值计算出深度图
    void pixelInterpolation(float* vertices, int verticesNum, unsigned char* depthMap);

protected:
    int height;
    int width;

    // 缓存一个像素上下左右第一个有值的像素位置
    int* imgCache;
};

#endif //FALIB_FACE_DEPTH_MAP_H
