##  classify-leaves

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

## 补充知识

### [SAVING AND LOADING MODELS](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)

1. [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save): Saves a serialized object to disk. This function uses Python’s [pickle](https://docs.python.org/3/library/pickle.html) utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function.

   > *“Pickling”* is the process whereby a Python object hierarchy is converted into a byte streamPickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [1](https://docs.python.org/3/library/pickle.html#id7) or “flattening”; however, to avoid confusion, the terms used here are “pickling” and “unpickling”.

2. [torch.load](https://pytorch.org/docs/stable/torch.html?highlight=torch load#torch.load): Uses [pickle](https://docs.python.org/3/library/pickle.html)’s unpickling facilities to deserialize pickled object files to memory. This function also facilitates the device to load the data into (see [Saving & Loading Model Across Devices](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)).

3. [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict): Loads a model’s parameter dictionary using a deserialized *state_dict*. For more information on *state_dict*, see [What is a state_dict?](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

`state_dict` : dict， maps each  layer to its parameter tensor. only **layers with learnable parameters** (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) 

Optimizer objects (`torch.optim`) also have a *state_dict* -- hyperparameters

**save ** with .pt or .pth file extension

```python
torch.save(model.state_dict(), PATH)
```

**load**

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval() # set dropout and batch normlization layers to eval model
```

**save** with .ckpt

```
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```

**load**

```
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

### Pandas

**pandas.DataFrame** :wo-dimensional, size-mutable, potentially heterogeneous tabular data.

dataframe.iloc([ a,b:c,d ])分区

### dict

python字典遍历：dict.items()

### TRANSFORMING AND AUGMENTING IMAGES

- module:`torchvision.transforms`  

- accept both PIL imges and tensor(B, C, H, W) images   (refer [docs](https://pytorch.org/vision/stable/transforms.html) for more detail)
- in validation and test, only use `transforms.Resize((224,224))` and `transforms.ToTensor()`

