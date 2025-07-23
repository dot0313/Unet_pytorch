# Unet_pytorch
一种基于U-Net的微观结构图像应力应变曲线预测模型（A U-Net-Based Model for Predicting Stress-Strain Curves from Microstructural Images）

## 简介

本项目使用 PyTorch 搭建了一个基于 U-Net 编码器的回归网络，用于从材料的微观结构图像预测应力-应变曲线。支持自定义数据集加载与训练。

## 软件架构
Python 3.x

PyTorch >= 1.10

其他依赖库：numpy, matplotlib, scikit-learn

## 安装教程
安装最新版的 PyTorch（ https://pytorch.org/ ）

安装必要的依赖项：

```pip install numpy matplotlib scikit-learn```

## 使用说明

当前数据集尚未提供，待准备完成后可按如下方式组织文件并运行训练脚本：

微观结构图像文件应为文本，形如：

```./1/Case1_Int1_Microstructure.txt```

应力-应变曲线文件应与图像一一对应，文件名形如：

```./1/Case1_Int1_Load.txt```

多个数据目录支持按编号（例如：./1, ./2）统一存放。

运行训练脚本：

```python U-Net_alldata_test1.py```

模型参数和预测图像将自动保存到以下目录：

验证集预测结果：./validation_alldata/

测试集预测结果：./test_alldata/

损失曲线图：loss_curve.png

## 注意事项

训练图像应为大小一致的二维数组，代码默认大小为 128x128。

预测输出为归一化后的应力值曲线，已在代码中自动还原为物理量。

文件配对需保证图像和应力数据文件同名结构（一个Microstructure.txt对应一个Load.txt）。

若未找到匹配曲线文件，将跳过该图像。
