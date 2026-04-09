
import os
import mxnet as mx
import numpy as np
from PIL import Image


class RecToImageSaver:
    def __init__(self, path_imgrec, path_imgidx, output_dir):
        # self.root_dir = root_dir
        self.output_dir = output_dir

        # # Define paths for record and index files
        # path_imgrec = os.path.join(root_dir, 'train.rec')
        # path_imgidx = os.path.join(root_dir, 'train.idx')

        # Load the record
        self.record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        # Read the image indices
        s = self.record.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.record.keys))
        print('Record file length:', len(self.imgidx))

        # Create output directory if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_images(self, cnt = -1): # -1 就是所有
        num = 0
        for idx in self.imgidx:
            num += 1
            if cnt > 0 and num > cnt:
                print(f"break for {cnt}")
                break
            # Read the image data from the record
            s = self.record.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            label = header.label

            # Decode the image using mxnet
            sample = mx.image.imdecode(img).asnumpy()

            # Convert to PIL image (RGB)
            sample = Image.fromarray(np.asarray(sample)[:, :, ::-1])

            # Save the image with a unique filename
            sample_save_path = os.path.join(self.output_dir, f"sample_{idx}.jpg")
            sample.save(sample_save_path)
            print(f"Saved image {sample_save_path} with label {label}")


if __name__ == "__main__":

    # 输入两个文件  rec idx  的路径
    path_imgrec = '/supercloud/llm-code/mkl/dataset/face/public/tmp_train/train.rec'  # Directory containing train.rec and train.idx
    path_imgidx = "/supercloud/llm-code/mkl/dataset/face/public/tmp_train/train.idx"
    output_dir = '/supercloud/llm-code/mkl/dataset/face/public/tmp/rec2img2'
    # 定义输出目录
    os.makedirs('/supercloud/llm-code/mkl/dataset/face/public/tmp/rec2img2', exist_ok=True)

    # Create the RecToImageSaver instance and save images
    rec_to_image_saver = RecToImageSaver(path_imgrec, path_imgidx, output_dir)
    rec_to_image_saver.save_images(cnt = 100) # 只保存前一百个看下
    print("done.")