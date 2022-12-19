# 介绍
地址：https://www.kaggle.com/competitions/amex-default-prediction/overview

> 本次竞赛由美国运通American Express发起的，属于风控比赛。在本次比赛中，将运用机器学习技能来预测信用违约，利用工业规模的数据来构建机器学习模型。训练、验证和测试数据集包括时间序列行为数据和匿名客户特征信息。
可以从创建特征到在模型中以更有机的方式使用数据，自由探索任何技术来创建最强大的模型。
具体来说，本次比赛的目的是根据客户每月的客户资料预测客户未来不偿还信用卡的概率。
目标二元变量是通过观察最近一次信用卡账单后18个月的绩效窗口来计算的，如果客户在最近一次账单日后的 120 天内未支付到期金额，则将其视为违约事件。

- 目的：根据客户每月的客户资料预测客户未来不偿还信用卡的概率
- 如何定义违约：如果客户在最近一次账单日后的 120 天内未支付到期金额，则将其视为违约事件。

## 数据

### 特点
- 总共50G，训练集10+G，测试集30+G
- csv格式
- 存在主办方均匀插入的噪声
- 该数据集包含每个客户在每个报表日期的汇总配置文件特征
- 特征通过了匿名和归一化处理，特征可以分为以下类别：
![img.png](https://github.com/ZhendongTang/Kaggle-American-Express-Default-Prediction/edit/main/img.png)

### Tips
- csv占存储空间，对数据进行压缩降低精度
- 条件有限，在有限的机器性能下把模型跑出来，分批读，把结果拼起来
- 使用论坛上去除了噪声的数据集

## 评价指标(由官方提供)：
M = 0.5（G+D)
- 规范化的基尼系数(G): 规范化的基尼系数= 2 * AUC-1，取值在-1和1之间， 如图所示。红色区域越大越好得分。
- 捕捉4%的违约率(D)：为一个全量样本总数的4%真正的阳性率(召回率) 。它对应于y坐标之间的交集绿线和红色roc曲线(绿点),总是在0和1之间。交点越高,更好的分数。
- 综上所述，竞争指标M衡量的是规范化的基尼系数G和捕捉4%的违约率D的平均水平。直观来说, 让我们同时优化暗红色曲线下面积（越大越好）和暗红色曲线与绿线的交点（越高越好）。
评价函数由两部分组成，第一部分权重默认为0.5，关注对正负样本的识别能力，第二部分权重是0.5，关注于对捕获率

为什么是0.5？ 
由主办方提供，可能是根据具体业务场景所得经验

![img_1.png](https://github.com/ZhendongTang/Kaggle-American-Express-Default-Prediction/edit/main/img_1.png)

## 模型选择
对于银行来说，可解释性要求高。故很少用深度的模型
结构化数据NN效果一般，先尝试使用树模型如lgbm
最终选择两个lgbm和一个xgb

## 特征工程
### 基于行为的聚合特征
### 同比环比特征
### 基于含义的延伸特征
#### 静态变动态
#### 跨期相减

```angular2html
- 删除特征 D_103、D_139（https://www.kaggle.com/code/raddar/redundant-features-amex/notebook）
- 对数值型变量添加 mean、min、max和std等统计特征
- 对spend_p+payment_p+delq+balance_p等特征计算sum构造新的特征
- 计算 'P-S-B'类特征，即对他们的和作差构造特征
- 构造first-last特征，即对客户最初和最后的状态作差和做除法
- 增加波动特征，即对特征的变动间隔1，2，3个状态作差（这个特征加入后,NN效果变差）
- 计算 After pay特征
- 构造客户状态周末计数特征
- 添加每月一周的相关信息特征
- 对类别特征进行了count、nunique、std、first做了聚合特征，且对其衍生了std,mean,min,max统计特征
- 对每个客户customer_ID和相同日期进行计数，创建消费次数特征
- 对于多个客户状态的客户计算客户“中间状态”特征，因为我们假设这将帮助我们涵盖客户的大部分变化
- 计算lag_features，即对部分特征的最小最大值做减法和除法
- P2B9特征构建(不是很理解，社区讨论说有效果)
- 对数值特征求本身减去、除以均值，对部分特征的最大值和最小值做减法和除法（除法是非线性可能对树模型有一定效果）
```

## 模型融合
(✔)加权融合 
`xgb0['prediction']*0.2 + lgb1['prediction']*0.4 +lgb2['prediction']*0.4`
> 排序融合 <br>  
加权排序融合 <br>  
stacking



## 亮点
- 特征工程（重要）
- 模型集成

## 不足
- 特征不够精细
- 融合策略比较粗糙
- NN探索不够

## 总结
- 第三名只用树模型，特征工程做了5000+特征，看社区讨论，学习别人的特征工程
- 探索NN

```angular2html
优秀方案推荐
https://github.com/JEddy92/amex_default_kaggle（有代码）
https://www.kaggle.com/code/pavelvod/27-place-sequentialencoder
https://www.kaggle.com/code/kalelpark/amex-top10-solution
```

# 运行步骤
- 下载原始数据集保存到input/amex-default-prediction目录；<br>  
  下载处理后的数据集到input/amex-data-integer-dtypes-parquet-format目录 <br>  
  主办方加了均匀分布的噪声，这个数据集去除了这些噪声，效果更好。（https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format）
- 运行code/fe_process.py，该步骤主要目的是生成特征文件，运行需要一定时间
- 运行code/lgb.py，这个代码训练lgb
- 运行code/xgb.py，这个代码训练xgb(使用GPU)
- 运行code/lgb_2.ipynb，这个代码训练第二个lgb
- 运行code/infer.ipynb，得到融合结果
- 提交sub/submission.csv文件到kaggle

# 依赖的包
- pandas
- numpy
- lightgbm
- pyarrow
- pickle
- tdqm

# Tips 
运行需要内存较大，建议64G内存
