//
// Created by EDY on 2022/5/19.
//

#ifndef LIBAG_NORMAL_CUES_H
#define LIBAG_NORMAL_CUES_H


class NormalCues {
public:
    NormalCues(int alignSize, int n, float normalizePercent = 0.95f);

    virtual ~NormalCues();

    // 返回法线信息图最小像素占比
    unsigned char *calNormalCuesMap(int *screenColors, unsigned char *iImage, int cutSize, int normalizeSize, unsigned char *toImage);

private:
    void
    calNormalCues(unsigned char *brightImage, unsigned char *darkImage, unsigned char *iImage, unsigned char *toImage,
                  int screenColor, int cutSize, int normalizeSize);

protected:
    int alignSize;
    int n;
    float normalizePercent;
    // 缓存像素差
    unsigned char *cache;
};


#endif //LIBAG_NORMAL_CUES_H
