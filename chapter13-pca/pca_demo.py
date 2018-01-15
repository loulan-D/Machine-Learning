# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:48:54 2018

@author: loulan
"""

# 鸢尾花数据分类

import matplotlib.pyplot as plt # 数据可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris    # 加载数据集

data = load_iris()   # 以字典形式加载鸢尾花数据集
y = data.target   #  y 数据集中的标签
X = data.data     # x 数据集中的属性数据
pca = PCA(n_components=2)    # 加载PCA算法，设置降维后主成份为2
reduced_X = pca.fit_transform(X)   # 对原始数据进行降维，保存到reduced_X中

# 按类别对降维后的数据进行保存
red_x,red_y = [],[]   # 第一类数据点
blue_x,blue_y = [],[]  # 第二类数据点
green_x,green_y  = [],[]  # 第三类数据点

# 按照鸢尾花的类别将降维后的数据点保存在不同的列表中
for i in range(len(reduced_X)):
    if y[i]==0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i]==1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

# 数据可视化
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
        
        
        
        
        
        
        
        