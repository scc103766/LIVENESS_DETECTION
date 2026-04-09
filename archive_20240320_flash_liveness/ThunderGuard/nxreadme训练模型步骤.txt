

训练出：闪光模型
训练：代码文件夹（pytg） 数据文件夹(data)

步骤：
1）虚拟环境（xxx）
E:\20240320闪光活体归档\environment

2）训练数据集（输入和标签）
dataset？
E:\20240320闪光活体归档\dataset\tg_export
修改：位置文件
把tg_export下train和test目录 copy 到pytg 上级的 data文件夹\sample的目录下

3）训练

```shell
cd pytg
python -m torch.distributed.launch --nproc_per_node <卡数> train.py --network MTGAN


4）模型文件
生成在E:\20240320闪光活体归档\ThunderGuard\resources
生成好的，导出。


5）评估和测试。
cd pytg
python test.py --network MTGAN


