# 人脸检测

## 1、将视频和它的描述文件取样成图

输入目录结构如下：
```text
video_path
    video_path/a.avi
    video_path/a.txt
    video_path/b.avi
    video_path/b.txt
# 视频根目录
    # 视频文件
    # 视频每一帧的描述
```

输出目录结构如下：
```text
# 输出取样的图片和图片对应的描述
img_path
    img_path/a
        img_path/a/1.jpg
        img_path/a/2.jpg
        ....
    img_path/a.txt
```

```shell
python sample_avi_2_png.py --frmo_path <from_video_path> --to_path <to_img_path>
```

## 在取样的图片对应的描述文件中的第二列加上人的双目坐标

```shell
python face_detect_with_landmark.py --data_path <to_img_path>
```