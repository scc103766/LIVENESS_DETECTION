# 人脸对齐工具

将同一个目录下，同一个人的多张人脸照片切割并对齐；同时在透明通道生成深度图。

系统要求：

* python3.6
* tensorflow<1.9



## 一、初步对齐

可以使用face_detect提供的工具实现如下功能：

* 取样视频成图
* 生成取样图的人眼坐标，以便初步对齐




## 二、像素级别对齐

### 1、编译linux下的对齐工具

```shell
cd falib
mkdir build
cd build
cmake -DUSE_PYTHON=ON ..
make
```

### 2、开始对齐，同时在第四个通道生成深度图

注意，如果要调整人脸框的大小自己看代码，调用时传相关的参数就行，代码注释有写

```shell
cd pyfa
python face_align.py --from_path ../data --to_path ../align_data
```

## 三、Android编译

```shell
cd falib
./build_android.sh or ./build_android_64.sh
```