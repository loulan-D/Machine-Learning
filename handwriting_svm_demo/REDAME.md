### 机器学习实现手写体字符识别 SVM+KNN

1. 数据集介绍
2. 选择哪种机器学习算法
3. 识别结果

----
#### 1. 数据集介绍
共5个字符，分别为M、N、P、O、5
他们的类别对应如下：
- M: 1
- N: 2
- P: 3
- O: 4
- 5: 5

图片像素大小与训练集测试集划分：
- 用画图工具画出这5个字符仿现实中字符
- 每个字符共有100张不同的手写图片，一共500张图片;
- 图片像素大小为50x50;
- 训练集和测试集划分为：8：2

数据集文件名解释：
- Sample1/1_1.png
- Sample1/1_2.png
- ….
- Sample1/1_2.png 中Sample后面的1为类别，后面的1_2，’_‘前面的1为类别，后面的2代表图片序号

------
#### 2. 选择SVM（支持向量机）机器学习算法
有关SVM的介绍 参考[SVM（支持向量机）](https://dongjingwei.wordpress.com/2019/07/15/svm-%e6%94%af%e6%8c%81%e5%90%91%e9%87%8f%e6%9c%ba/)


-----

#### 3. 结果
- train_dataset data shape: (400, 2500)
- train_dataset table shape: (400,)
- test_dataset data shape: (100, 2500)
- test_dataset table shape: (100,)
- 预测结果:
['5' '4' '5' '5' '5' '5' '5' '4' '5' '5' '5' '5' '4' '5' '4' '5' '4' '5'
 '5' '5' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '3' '4'
 '5' '4' '4' '4' '3' '3' '3' '3' '3' '3' '3' '3' '3' '3' '4' '3' '3' '4'
 '3' '3' '3' '3' '4' '3' '2' '4' '2' '2' '2' '2' '2' '2' '2' '1' '2' '2'
 '2' '1' '2' '1' '2' '2' '2' '1' '4' '4' '4' '4' '4' '4' '4' '4' '1' '1'
 '4' '4' '1' '4' '1' '1' '4' '4' '4' '4'];
- 真实结果:['5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5' '5'
 '5' '5' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4'
 '4' '4' '4' '4' '3' '3' '3' '3' '3' '3' '3' '3' '3' '3' '3' '3' '3' '3'
 '3' '3' '3' '3' '3' '3' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2' '2'
 '2' '2' '2' '2' '2' '2' '2' '2' '1' '1' '1' '1' '1' '1' '1' '1' '1' '1'
 '1' '1' '1' '1' '1' '1' '1' '1' '1' '1']

- SVM Model Score:0.7
- KNN n_neightbors: 1 weight: uniform score: 0.85
- KNN n_neightbors: 1 weight: distance score: 0.85
- KNN n_neightbors: 2 weight: uniform score: 0.85
- KNN n_neightbors: 2 weight: distance score: 0.85
- KNN n_neightbors: 3 weight: uniform score: 0.84
- KNN n_neightbors: 3 weight: distance score: 0.84
- KNN n_neightbors: 4 weight: uniform score: 0.85
- KNN n_neightbors: 4 weight: distance score: 0.85
- KNN n_neightbors: 5 weight: uniform score: 0.84
- KNN n_neightbors: 5 weight: distance score: 0.84
- KNN n_neightbors: 6 weight: uniform score: 0.84
- KNN n_neightbors: 6 weight: distance score: 0.84