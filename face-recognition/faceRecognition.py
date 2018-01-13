# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:18:45 2018

@author: loulan
"""

import numpy as np
import os
import operator
import matplotlib.pyplot as plt
from PIL import Image         # 图像处理
import scipy.misc as sm       # 图像处理


"""
把图像拉成一个长向量，然后pca降维作为特征
距离函数，用欧式距离
每类n个图像做训练，其余的做测试，讨论训练样本数量对识结果的影响
5：5    6：4     7：3      8：2
1.读取图像，做pca，变成特征文件
2.写距离函数描述人脸的相似性
3.knn分类
-------------
一、随机选7副训练，3副测试，然后实验重复30次以上，统计平均识别率
二、做cv，交叉验证
"""


#  PCA 主成成分分析 选取特征值大的特征向量作为主成分
def pca(data,k):                   # data为训练样本   data 就是样本集  低维空间的维度m
    data = np.float32(np.mat(data)) 
    rows,cols = data.shape        # 训练样本的维度
    data_mean = np.mean(data,0)    #对列求均值          # 求出一个平均脸
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
    return data_new,data_mean,V1  # 降维后的数据，即作为训练已经提取特征向量后的数据 平均脸 ，归一化之后的特征向量V1


### 将每一副图像拉平，降成一维图像
def img_vector(filename):              
#    img = Image.open(filename).convert('L')   # 原始图片的分辨率为 (112,92)
#    m,n = np.array(img).shape[:2]
#    vector  = np.reshape(img,(1,m*n))
    img = sm.imread(filename)
    vector = img.flatten()
    return vector                     # 将图片拉平  现在是 (1,10304)


### 加载图像数据集并选择训练样本个数
def load_data_set(k):  #选择每个人的10副图像中 k 个人作为训练数据    
    data_path = r"D:\task\12月\机器学习\orl"       # orl 人脸库文件夹
    
    choose = np.random.permutation(10)+1        # 对每个人的10副图像 1 -10 生成一个随机的序列
    train_face = np.zeros((40*k,112*92))        # 训练数据矩阵，40*k 行  10304列
    train_face_number = np.zeros(40*k)          # 训练数据
    test_face = np.zeros((40*(10-k),112*92))    # 测试数据矩阵，40*（10-k）行，10304列
    test_face_number = np.zeros(40*(10-k))      # 测试数据
    for i in np.arange(40):                     #40个样本，每个样本10副图像
        people_num = i+1
        for j in np.arange(10):                 #每个人有10副图像
            if j < k:
                filename = data_path +'\\s'+str(people_num)+'\\'+str(choose[j])+'.pgm'
                img = img_vector(filename)     
                train_face[i*k+j,:] = img
                train_face_number[i*k+j] = people_num
            else:
                filename = data_path +'\\s'+str(people_num)+'\\'+str(choose[j])+'.pgm'
                img = img_vector(filename)     
                test_face[i*(10-k)+(j-k),:] = img
                test_face_number[i*(10-k)+(j-k)] = people_num

    return train_face,train_face_number,test_face,test_face_number

# 计算测试数据的正确率
def face_recog(k,m): 
    # 得到训练样本和测试样本数据集
    train_face,train_face_number,test_face,test_face_number = load_data_set(k)
    # 用pca训练样本，
    data_train_new,data_mean,V = pca(train_face,m)      #训练后降维的训练数据 , m为特征向量的个数
    num_train = data_train_new.shape[0]
    num_test = test_face.shape[0]                    # 40*（10-k）个
    temp_face = test_face - np.tile(data_mean,(num_test,1))    # 对测试人脸的样本进行中心化处理
    data_test_new = temp_face*V                          #得到测试脸在特征向量下的数据
    data_test_new = np.array(data_test_new)              # 测试脸ndarray
    data_train_new = np.array(data_train_new)            # 训练人脸在特征向量下的数据
    true_num = 0      # knn 分类，欧式距离
    for i in np.arange(num_test):
        testFace = data_test_new[i,:]
        diffMat = data_train_new - np.tile(testFace,(num_train,1))
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        sortedDistIndicies = sqDistances.argsort()    # argsort函数返回的是数组值从小到大的索引值
        indexMin = sortedDistIndicies[0]
        if train_face_number[indexMin] == test_face_number[i]:
            true_num += 1

    accuracy = float(true_num)/num_test
    print ('识别的正确率: %.2f%%'%(accuracy * 100))
    return accuracy*100

# 平均识别率
def average(k,m):        # k:训练样本数  m:特征向量
    sum = 0
    for i in range(30):    # 30次
        train,train_number, test,test_number  = load_data_set(k)
        accuracy = face_recog(k,m)
        sum += accuracy
    print("特征向量取前{}个".format(m))
    print("一共测试30次,平均识别率是%.2f%%"%(sum/30))
    print("<------------------>")
    return sum/30
    
if __name__  == '__main__':
    k = int(input("10副图像中选择多少副作为训练数据"))
    m = int(input("选取多少个特征向量作为主成分"))
    train,train_number, test,test_number  = load_data_set(k)
    print("40*10一共400张图片，其中训练样本 {} 张,测试样本 {} 张".format(train_number.shape[0],test_number.shape[0]))
    face_recog(k,m)
    
    # 计算平均识别率
#    averages = []
#    for i in range(10,21):
#        averages.append(average(9,i))
#    plt.plot(np.arange(10,21),averages)
#    plt.xlabel('tezheng')
#    plt.ylabel('accurages')
#    plt.title('7:3-----')
#    plt.show()
    
        
    
    
