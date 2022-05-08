# Chinese-Text-Classification-Pytorch
本项目基于 FAST IS BETTER THAN FREE:
REVISITING ADVERSARIAL TRAINING的论文复现，主要复现了Free，FGSM，PGD三种算法在文本分类上的效果

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万

## 代码解析
由于论文原文是直接添加delta到Image，在文本分类中需要添加到embedding上，因此代码部分为：
```
 def forward(self, x, delta=None):
        out = self.embedding(x[0])
        if delta is not None:
            out += delta
```
论文主要引入了3个超参数： alpha, attack_iters以及epsilon来控制delta的初始值范围，根据原论文实验，我们可以发现 对FGSM来说alpha的取值最好在1~2 epsilon之间，论文使用了1.25，在对比实验里没有搜索最优超参数，而是设置alpha=epsilon， 也不微调attack_iters，所有实验的超参数为
```
'epsilon': 0.01,
'alpha': 0.01,
'attack_iters': 5,
```
在实验中，尝试了对delta的不同初始化方法，其中0初始化效果极差，因此都使用uniform_初始化方法。 
 
## 使用说明
```
# 训练并测试：
# TextCNN baseline
python run.py --model TextCNN --training_mode CE

# TextCNN pgd
python run.py --model TextCNN --training_mode pgd

# TextCNN fgsm
python run.py --model TextCNN --training_mode fgsm

# TextCNN free
python run.py --model TextCNN --training_mode free
```

## 模型效果
| **Model** | **Accuracy** | **Precision**|**Recall** |**F1-score**|
| ------| ------| ------| ------| ------|
|TextCNN|	90.45	|90.52|	90.45	|90.46|
|FGSM|	91.73	|91.73	|91.73|	91.71|
|PGD	|91.43|	91.46|	91.43|91.41|
|FREE	|91.06	|91.11|	91.06	|91.06|


四种模型训练消耗的时间(minutes)对比如下：
|**model**| **total_cost**|	
| ------| ------| 
|TextCNN|	1.57	|
|FGSM|	8.47|	
|PGD|	20.43	|
|FREE	|4.09|	

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。

6 结论
=
 + 对抗训练技术方法的确有助于提高文本分类任务的效果；<br>
 + 实验表明随机初始化delta就可以让FGSM的效果好于PGD和Free，但是这并不代表FGSM在别的数据集上效果也很好；<br>
 + 由于超参数比较多，对抗学习的训练是Data-driven的，需要合理设置超参数；<br>
 + 可以考虑一种两阶段训练的方法和组合优化的方法来优化；<br>


## Reference
[1] Convolutional Neural Networks for Sentence Classification  
[2] FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING  
[3] https://github.com/649453932/Chinese-Text-Classification-Pytorch <br>
[4] https://github.com/locuslab/fast_adversarial
