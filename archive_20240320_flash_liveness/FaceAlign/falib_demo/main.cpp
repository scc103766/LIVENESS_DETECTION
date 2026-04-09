#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "nx_fa_export.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

using namespace std;

vector<vector<int>> readParmList(const string &path, bool isX) {
    vector<vector<int>> parmList;
    ifstream infile(path.data());
    while (infile){
        string allLine;
        if (!getline(infile, allLine)) break;
        istringstream allLineReader(allLine);
        if(allLineReader){
            string subLine;
            getline(allLineReader, subLine, ';');
            // 如果是读取y轴
            if(!isX){
                getline(allLineReader, subLine, ';');
            }
            istringstream subLineReader(subLine);
            vector<int> c;
            while (subLineReader){
                string s;
                if (!getline(subLineReader, s, ',')) break;
                c.push_back(atoi(s.c_str()));
            }
            parmList.push_back(c);
        }
    }
    return parmList;
}

int main() {
    // 初始化---------------------------------------------------------------------------
    FA_Init_Parameter initParameter;
    initParameter.height = 1920;
    initParameter.width = 1080;
    initParameter.frameNum = 9;
    initParameter.alignSize = 384;
    initParameter.faceSize = 208;
    initParameter.cutSize = 256;
    initParameter.maxAlignOffsetPixel = 4;
    initParameter.maxKeyPointsNum = 70;
    initParameter.leftEyeId = 68;
    initParameter.rightEyeId = 69;
    initParameter.normalizePercent = 0.95f;
    initParameter.mergeWeight = 1;
    FA_Handle *pHandle;
    FA_Initial(&pHandle, initParameter);
    cout<<"完成初始化\n";

    // 一帧一帧插入图片---------------------------------------------------------------------
    for(int i = 0; i < initParameter.frameNum; ++i){
        std::string imgPath = "../resources/1_1658973556881_yy_1/" + std::to_string(i+1) + ".jpg";
        cv::Mat img = cv::imread(imgPath, -1);
        FA_Insert_Parameter insertParameter;
        insertParameter.frameId = i;
        insertParameter.img = img.data;
        FA_InsertImage(pHandle, insertParameter);
    }

    // 插入完图片后可以做关键点检测--------------------------------------------------------------
    for(int i = 0; i < initParameter.frameNum; ++i){
        auto img = FA_GetCurrentInsertImage(pHandle, i);
    }
    // 我将人脸检测得到的关键点预存在文件里面，所以demo不需要人脸关键点检测。客户端需要自己对关键点做检测
    std::string keyPointPath = "../resources/1_1658973556881_yy_1_key_point.txt";
    auto keyPointX = readParmList(keyPointPath, true);
    auto keyPointY = readParmList(keyPointPath, false);
    cout<<initParameter.frameNum<<"="<<keyPointX.size()<<"="<<keyPointY.size()<<endl;
    cout<<initParameter.maxKeyPointsNum<<"="<<keyPointX[0].size()<<"="<<keyPointY[0].size()<<endl;

    // 开始插入
    FA_KeyPoints_Parameter keyPointsParameter;
    keyPointsParameter.keyPointsX = new int[initParameter.maxKeyPointsNum];
    keyPointsParameter.keyPointsY = new int[initParameter.maxKeyPointsNum];
    for(int i = 0; i < initParameter.frameNum; ++i){
        keyPointsParameter.frameId = i;
        for(int j = 0; j < initParameter.maxKeyPointsNum; ++j){
            keyPointsParameter.keyPointsX[j] = keyPointX[i][j];
            keyPointsParameter.keyPointsY[j] = keyPointY[i][j];
        }

        FA_SetCurrentInsertKeyPoints(pHandle, keyPointsParameter);
    }

    // 利用插入的人脸检测关键点信息，做人脸对齐---------------------------------------------------------
    cout<<"开始对齐\n";
    cout<<FA_FinishInsertImage(pHandle)<<endl;
    // 获取对齐后的图片
    auto alignImage = FA_GetAlignFace(pHandle);
    cv::Mat alignMat = cv::Mat(initParameter.cutSize*initParameter.frameNum, initParameter.cutSize, CV_8UC3, alignImage);
    cv::imwrite("alignMat.jpg", alignMat);

    // 退出前别忘释放空间----------------------------------------------------------------------------
    delete []keyPointsParameter.keyPointsX;
    delete []keyPointsParameter.keyPointsY;
    FA_Uninitial(pHandle);
    return 0;
}
