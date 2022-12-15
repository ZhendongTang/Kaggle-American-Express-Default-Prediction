运行步骤
1 下载原始数据集保存到input/amex-default-prediction目录；
下载处理后的数据集到input/amex-data-integer-dtypes-parquet-format目录
（https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format）
2 运行code/fe_process.py，该步骤主要目的是生成特征文件，运行需要一定时间
3 运行code/lgb.py，这个代码训练lgb
4 运行code/xgb.py，这个代码训练xgb(使用GPU)
5 运行code/lgb_2.ipynb，这个代码训练第二个lgb
6 运行code/infer.ipynb，得到融合结果
7 提交sub/submission.csv文件到kaggle



一些依赖的包
pandas
numpy
lightgbm
pyarrow
pickle
tdqm

运行需要内存较大，建议64G内存
