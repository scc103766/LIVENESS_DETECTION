//
// Created by Yanyu on 2022/5/23.
//

#ifndef TG_API_TG_EXPORT_H
#define TG_API_TG_EXPORT_H

#ifdef WIN32 // or something like that...

#ifdef _USRDLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#else
#define DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // 程序句柄
    struct TG_Handle;

    // 初始化参数
    struct TG_Init_Parameter{
        // 人脸框标记图片
        unsigned char* mask;
        // 图片的高
        int height;
        // 图片的宽
        int width;
        // 闪光数量
        int flashNum;
        // 对齐图的大小，默认480
        int alignSize;
        // 归一化像素比例，默认设置0.95
        float normalizePercent;
    };

    /*
     * 初始化程序：
     * 返回值：
     *  0：成功
     *  1：目前只支持5次闪光
     *  2：图片尺寸太小
     * */
	DLLEXPORT int TG_Initial(TG_Handle **ppHandle, TG_Init_Parameter initParameter);

    // 获取版本号
	DLLEXPORT const char* TG_GetVersion();

    // 释放程序资源，返回0表示成功
	DLLEXPORT int TG_Uninitial(TG_Handle *pHandle);

    // 插入图片的参数
    struct TG_Insert_Parameter{
        // 插入的一帧图片
        unsigned char *img;
        // 当前闪屏颜色
        int screenColor;
        // 传入的状态
        //- state=0：第一个白屏和第五个黑屏时的第一帧图片，需要将state设置成0
        //- state=1：第二、三、四个闪屏时的第一帧图片，需要将state设置成1
        //- state=2：第一个白屏时的非第一帧，需要将state设置成2
        //- state=3：第五个黑屏时的非第一帧，需要将state设置成3
        //- state=4：第二、三、四个闪屏时的非第一帧，需要将state设置成4
        int state;
        // 默认设置成2，示每隔两个像素核对一下图片是否合适，这样就不用每个像素都遍历一次
        int step;
        // 图片编码
        //- encode=0：表示 bgr
        //- encode=1：表示 yuv420
        int encode;
        // 将mask图顺时针旋转rotate度，可以和img对上。rotate=[0, 90, 180, 270]
        int rotate;
    };

    /*
     * 判断:
     * 返回值：
     *  0：需要
     *  1：插入太多图片
     *  2：插入的颜色不对
     *  3：不满足插入条件
     *  4：编码不支持
     * */
	DLLEXPORT int TG_IsInsertImage(TG_Handle *pHandle, TG_Insert_Parameter insertParameter);


    /*
    * 结束插入图片:
    * 返回值：
    *  0：成功
    *  1: 之前核对的照片不够
    * */
    DLLEXPORT int TG_CalNormalCues(TG_Handle *pHandle, unsigned char* alignImages);

    // 获取加密过的法线信息图
	DLLEXPORT unsigned char* TG_GetNormalCues(TG_Handle *pHandle);


#ifdef __cplusplus
}
#endif

#endif //TG_API_TG_EXPORT_H
