这是[kaggle树叶分类比赛](https://www.kaggle.com/competitions/classify-leaves)的解答

## Preprocess

### Pandas

pandas是一个CSV文件处理库。比赛中提供的训练CSV文件包含图片相对路径和标注类别，我们需要对类别进行去重并编码为数字，才能给到模型进行训练。

- 去重：通过set()去重，sorted()排序
- 编码：对上述list zip为dict，实现类名和数字的映射

### LeavesDataset

该类继承自torch.Dataset

- `__init__`( ) :初始化目录，mode ，依据比例对train.csv进行 train, val分区
- `__getitem__()`: 确定数据集中所返回那一部分数据。Image.Open读入图片，划分标签区域不同mode下返回不同行为
  - train ： resize, 数据增广， totensor，返回img， label_num
  - val/test：resize, totensor, 返回img

- 最后通过DataLoader对DataSet划分batch_size, 是否shuffle

## Model

torchvision 中有已经定义好的model，改掉最后一层fc就行

## How to Train

训练过程我们需要初始化：

- 超参数，比如：lr，weight_decay， device，epoch
- 优化器
- loss

训练流程如下：

```
每一个epoch下每一个epoch下：
	train_dataloader中每一个batch下：
		得到image, label
		image, label放到device上
		image输入model得到输出
		对输出和label计算loss
		清空优化器梯度
		loss反向传播
		优化器迭代参数
		收集acc
	val_dataloader中的每一个batch下：
		得到image, label
		image, label放到device上
		在不计算梯度的情况下：
			image输入model得到输出
		对输出和label计算loss
		收集acc
	acc对比，看是否保存
```

## How to Predict

- 初始化模型结构
- 加载模型参数
- 设为eval状态
- 输入模型，取输出最大值，映射为类名
- csv处理

>补充csv内容

>补充 load_static_model

> 文件分类