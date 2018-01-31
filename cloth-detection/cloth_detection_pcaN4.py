# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:38:34 2018

@author: loulan
"""

import numpy as np
import scipy.misc as sm
import os
import time

# 原数据是纺织品布料检测
# 分别用 pca Hog  haar lbp 做特征提取
# 经向样本 72;  块状疵点  202;   纬向疵点  117;    正常疵点  398     训练样本  789
# 经向样本 177 ； 块状疵点 400；  纬向疵点 379； 正常疵点 386  测试样本一共 1342
# 训练样本和测试样本可以颠倒试验


# 加载图像，将图像拉成一维向量 
def imgVector(filename):
    img = sm.imread(filename)     #128*128
    vector = img.flatten()        # 每副图像拉平成一维的向量
    return vector    


#  训练样本  789， 测试样本  1342
# 加载训练数据集
def load_train_data():
#    trainfile = r"D:\task\12月\机器学习\128-128实验样本\No4\训练样本"  # 789 做训练
    trainfile = r"D:\task\12月\机器学习\128-128实验样本\No4\测试样本"  #1342  做训练
    listdir = os.listdir(trainfile)   # 训练数据集分类列表  [经向样本 块状疵点  纬向疵点  正常疵点]
    dirs = []
    for file in listdir:
        file = os.path.join(trainfile,file)
        dirs.append(file)
    
#    train_data = np.zeros((789,128*128))   # 0矩阵 (789,128*128)
#    train_data_number = np.zeros(789)      # 训练数据的样本个数
    
    train_data = np.zeros((1342,128*128))   # 0矩阵 (1342,128*128)
    train_data_number = np.zeros(1342)      # 训练数据的样本个数
    index  = 0
    differ = 0      # 分类  共分为4类样本
    for file in dirs:
        differ += 1
        for filename in os.listdir(file):
            filename = os.path.join(file,filename)
            img = imgVector(filename)
            train_data[index,:] = img      # 每一幅图像都置为大矩阵中的一个样本记录
            train_data_number[index] = differ    #同一类置为相同的数字
            index += 1
        
    return train_data,train_data_number     # 训练数据矩阵，每一幅训练数据的类别

# 加载测试数据集
def load_test_data():
#    testfile = r"D:\task\12月\机器学习\128-128实验样本\No4\测试样本"  # 1342 做测试
    testfile = r"D:\task\12月\机器学习\128-128实验样本\No4\训练样本"  # 789  做测试
    listdir = os.listdir(testfile)
    dirs = []
    for file in listdir:
        file = os.path.join(testfile,file)
        dirs.append(file)
    
#    test_data = np.zeros((1342,128*128))
#    test_data_number = np.zeros(1342)
    
    test_data = np.zeros((789,128*128))
    test_data_number = np.zeros(789)
    index  = 0
    differ = 0
    for file in dirs:
        differ += 1
        for filename in os.listdir(file):
            filename = os.path.join(file,filename)
            img = imgVector(filename)
            test_data[index,:] = img
            test_data_number[index] = differ
            index += 1
        
    return test_data,test_data_number

# 加载数据集     
def load_set():
    # 789 训练  1342 测试
#    train_data,train_data_number = load_train_data()  # (789,16384)  (789,)
#    test_data,test_data_number = load_test_data()     #(1342,16384)   (1342,)
    # 1342 训练  789 测试
    train_data,train_data_number = load_train_data()  # (1342,16384)  (1342,)
    test_data,test_data_number = load_test_data()     #(789,16384)   (789,)
    return train_data,train_data_number,test_data,test_data_number 


#  PCA 主成成分分析 选取特征值大的特征向量作为主成分
def pca(data,k):                   # data为训练样本   data 就是样本集  低维空间的维度k
    data = np.float32(np.mat(data)) 
    rows,cols = data.shape         # 训练样本的维度
    data_mean = np.mean(data,0)    #对列求均值          
    data_mean_all = np.tile(data_mean,(rows,1))      
    Z = data - data_mean_all                          # 对所有样本进行中心化 Z 是中心化之后的样本
    T1 = Z*Z.T                    #使用矩阵计算，所以前面mat   计算样本协方差矩阵
    D,V = np.linalg.eig(T1)       #特征值与特征向量     对协方差矩阵做特征值分解   V是特征向量
    V1 = V[:,0:k]                 #取前k个特征向量      所有行，取前k列作为特征向量
    V1 = Z.T*V1                  
    for i in np.arange(k):        #对特征向量V1进行归一化
        L = np.linalg.norm(V1[:,i])
        V1[:,i] = V1[:,i]/L
        
    data_new = Z*V1               # 降维后的数据
    return data_new,data_mean,V1  # 降维后的数据，即作为训练已经提取特征向量后的数据  ，归一化之后的特征向量V1

# 求预测的正确率
def accuracy_recog(k):     # k 保留的特征向量个数
    train_data,train_data_number,test_data,test_data_number = load_set()
    data_train_new,data_mean,V = pca(train_data,k)
    num_train = data_train_new.shape[0]
    num_test = test_data.shape[0]                  
    temp_face = test_data - np.tile(data_mean,(num_test,1))   
    data_test_new = temp_face*V                          
    data_test_new = np.array(data_test_new)              
    data_train_new = np.array(data_train_new)            
    true_num = 0      # knn 分类，欧式距离
    for i in np.arange(num_test):
        testFace = data_test_new[i,:]
        diffMat = data_train_new - np.tile(testFace,(num_train,1))
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        sortedDistIndicies = sqDistances.argsort()    # argsort函数返回的是数组值从小到大的索引值
        indexMin = sortedDistIndicies[0]
        if train_data_number[indexMin] == test_data_number[i]:
            true_num += 1

    accuracy = float(true_num)/num_test
    print("一共有{}张图片做训练，有{}张图片做测试".format(train_data_number.shape[0],test_data_number.shape[0]))
    print("特征向量取前{}个".format(k))
    print ('识别的正确率: %.2f%%'%(accuracy * 100))
    return accuracy*100

# 平均识别率
def average():        
    sum = 0
    for i in range(25,45):    # 20次
        accuracy  = accuracy_recog(i)
        sum += accuracy
    print("<------------------>")
    print("一共测试20次,平均识别率是%.2f%%"%(sum/20))
    print("<------------------>")
    return sum/20
# No4 文件夹
# 789 张训练  1342 张测试 正确率 大约 84.43%
# 1342 张训练   789 张测试  正确率 大约 97.13%

if __name__ == '__main__':
    start = time.time()
    average()
    end = time.time()
    print("程序执行时间是{}".format(end-start))
    # 测试20次，程序执行时间是156.84046602249146
        
    
        
