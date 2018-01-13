# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:07:11 2018

@author: loulan
"""

# Machine Learning in Action 根据《机器学习实战》
"""
（knn）k-近邻算法工作原理：
存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，知道样本集中每一个数据与其所属分类的对应关系。
输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）
的分类标签。只选择样本数据集中前k个最相似的数据，这就是k的出处。最后，选择k个最相似数据中出现次数最多的分类，作为新数据
的分类。

k-近邻算法的优缺点：
优点：精度高，对异常值不敏感，无数据输入假定
缺点：计算复杂度高，空间复杂度高
适用范围：数值型和标称型

# knn 伪代码：
# 使用knn将某组数据划分到某个类当中
对未知类别属性的数据集中的每个点
1.计算已知类别数据集中的点与当前点之间的距离
2.按照距离递增次序排序
3.选取与当前点距离最小的k个点
4.确定前k个点所在类别的出现频率
5.返回前k个点出现频率最高的类别作为当前点的预测分类
"""

import numpy as np

# 创建数据集 group：每一个样本数据   labels: 分类
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['a','a','b','b']
    return group,labels

# knn
def classify0(inX,dataSet,labels,k):  # 预测数据，训练样本集，标签向量，选择最近邻的数目
    dataSetSize = dataSet.shape[0]    # 样本集的行数
    # np.tile(a,(b,c)) 通过重复给定的次数来构造函数  a,初始数组  b,重复的次数， c ,行数
    diffMat = np.tile(inX,(dataSetSize, 1)) - dataSet # 预测点与样本集中的每个点作差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5   # 利用欧式距离公式   [1.48,1.41,0,0.1]   [2,3,1,0] 这个返回的索引与label的索引相对应
    sortedDistIndicies = distances.argsort()  #argsort() 按照数组的某一列进行排序，返回排序后元素在该列的次序
    classCount= {}     # final result {'a':1,'b':2}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) +1
    # 对classCount字典进行排序，将value大的排在前面
    sortedClassCount = sorted(classCount.items(),key = lambda asd:asd[1],reverse = True)
    return sortedClassCount[0][0]     
        
if __name__ == '__main__':
    group,labels = createDataSet()
    inx = [0,0]
    final = classify0(inx,group,labels,3)
    print("{}属于{}类".format(inx,final))
