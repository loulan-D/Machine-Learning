# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:37:30 2018

@author: loulan
"""

import numpy as np

def loadDataSet(fileName,delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return np.mat(datArr)
    
def pca(dataMat,k=10):     # K个特征
    meanVals = np.mean(dataMat,axis = 0)
    meanRemoved = dataMat - meanVals     # 减去原始数据的平均值
    covMat = np.cov(meanRemoved,rowvar = 0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))# 计算协方差矩阵和特征值
    eigValInd = np.argsort(eigVals)     # 对特征值从小到大排序
    eigValInd = eigValInd[:,-(k+1):-1]   # 逆序后前k个最大的特征向量
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat*redEigVects.T)+meanVals    
    return lowDDataMat,reconMat 
    

if __name__ == '__main__':
    path = r'testSet.txt'
    dataMat= loadDataSet(path)
    lowDMat,reconMat = pca(dataMat)