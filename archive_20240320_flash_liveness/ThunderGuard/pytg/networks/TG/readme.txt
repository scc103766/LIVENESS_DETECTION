这是对原版论文修改
改动如下：
1、计算pixel wise loss的时候，让亮的像素有更大的权重
2、除了原版，其他的数据分布都改过了，加大复杂样本的取样概率
3、加入positionEmbedding