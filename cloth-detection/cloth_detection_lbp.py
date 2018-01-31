# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 23:43:42 2018

@author: loulan
"""

# 布料检测 目标检测的图像特征提取  LBP特征
# 读取图像 可以使用opencv 也可以使用skimage  
# different  读图 opencv  读进来是BGR ,而skimage 是RGB ； 显示 opencv 使用BGR ，而matplotlib.pyplot 使用RGB
"""
LBP（Local Binary Pattern 局部二值模式）是一种用来描述局部纹理特征的算子；用于纹理特征提取，而且提取的特征是图像的局部
的纹理特征
知识参考：http://www.cnblogs.com/ywsoftware/p/4434337.html
LBP特征的描述：
原始的LBP算子定义为在3*3的窗口内，以窗口中心像素为阈值，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于像素值，则该像素点的位置被标记为1，否则为0.这样，3*3邻域内的8个点经过比较可以产生8位二进制数（转化为十进制即为LBP码），这样就得到该窗口中心像素点的LBP值，并用这个值反映区域的纹理信息。
LBP检测的原理：
LBP算子在每个像素点都可以得到一个LBP“编码”，那么对一副图像（记录的是每个像素点的灰度值）提取原始的LBP算子后，得到的原始LBP特征依然是“一副图片”。

例如：一幅100*100像素大小的图片，划分为10*10=100个子区域（可以通过多种方式来划分区域），每个子区域的大小为10*10像素；在每个子区域内的每个像素点，提取其LBP特征，然后，建立统计直方图；这样，这幅图片就有10*10个子区域，也就有了10*10个统计直方图，利用这10*10个统计直方图，就可以描述这幅图片了。之后，我们利用各种相似性度量函数，就可以判断两幅图像之间的相似性了。

对LBP特征向量进行提取的步骤：
（1）首先将检测窗口划分为16×16的小区域（cell）；
（2）对于每个cell中的一个像素，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3*3邻域内的8个点经比较可产生8位二进制数，即得到该窗口中心像素点的LBP值；
（3）然后计算每个cell的直方图，即每个数字（假定是十进制数LBP值）出现的频率；然后对该直方图进行归一化处理。
（4）最后将得到的每个cell的统计直方图进行连接成为一个特征向量，也就是整幅图的LBP纹理特征向量；
然后便可利用SVM或者其他机器学习算法进行分类了

"""
# 用LBP方法提取纹理特征，用SVM做分类
import time
import os
import numpy as np
import scipy.misc as sm
from skimage import feature

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR 

# 读取图片
def imgVector(filename):
    img = sm.imread(filename)     #128*128
    return img

# 加载训练数据集
def load_train_data():
#    trainfile = r"D:\task\12月\机器学习\128-128实验样本\No4\训练样本"  # 789 做训练
    trainfile = r"D:\task\12月\机器学习\128-128实验样本\No4\测试样本"  #1342  做训练
    listdir = os.listdir(trainfile)   # 训练数据集分类列表  [经向样本 块状疵点  纬向疵点  正常疵点]
    dirs = []
    for file in listdir:
        file = os.path.join(trainfile,file)
        dirs.append(file)
    
#    train_data = np.zeros((789,128,128))   # 0矩阵 (789,128,128)
#    train_data_number = np.zeros((789))      # 标签
    
    train_data = np.zeros((1342,128,128))   # 0矩阵 (1342,128,128)
    train_data_number = np.zeros((1342))      # 训练数据的样本个数
    index  = 0
    differ = 0      # 分类  共分为4类样本
    for file in dirs:
        differ += 1
        for filename in os.listdir(file):
            filename = os.path.join(file,filename)
            img = imgVector(filename)
            train_data[index,:,:] = img      # 每一幅图像都置为大矩阵中的一个样本记录
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
    
#    test_data = np.zeros((1342,128,128))
#    test_data_number = np.zeros((1342))
#    
    test_data = np.zeros((789,128,128))
    test_data_number = np.zeros((789))
    index  = 0
    differ = 0
    for file in dirs:
        differ += 1
        for filename in os.listdir(file):
            filename = os.path.join(file,filename)
            img = imgVector(filename)
            test_data[index,:,:] = img
            test_data_number[index] = differ
            index += 1
        
    return test_data,test_data_number

# 加载训练和测试数据集
def load_set():
    train_data,train_data_number = load_train_data()
    test_data,test_data_number = load_test_data()
    
    return train_data,train_data_number,test_data,test_data_number

#http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern 
def lbp_detection():
    train_data,train_data_number,test_data,test_data_number = load_set()
    radius = 1;
    n_point  = radius * 8
#    train_hist = np.zeros((789,256))
#    test_hist = np.zeros((1342,256))
    
    train_hist = np.zeros((1342,256))
    test_hist = np.zeros((789,256))
#    for i in np.arange(789):
    for i in np.arange(1342):
        # 使用LBP方法提取图像的纹理特征。
        lbp = feature.local_binary_pattern(train_data[i],n_point,radius,method = 'default')
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist
        train_hist[i], _ = np.histogram(lbp,bins = max_bins,range = (0,max_bins),normed = True)
#        train_hist[i],_= np.histogram(lbp,range(0,max_bins),normed = True,bins = max_bins)
#    for i in np.arange(1342):
    for i in np.arange(789):
        lbp = feature.local_binary_pattern(test_data[i],n_point,radius,method = 'default')
        max_bins = int(lbp.max()+1)
        test_hist[i], _ = np.histogram(lbp,bins = max_bins,range = (0,max_bins),normed = True)

    return train_hist,test_hist;

def accuracy_svm():
    train_data,train_data_number,test_data,test_data_number = load_set()
    train_hist,test_hist = lbp_detection()
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1);
    accuracy = OneVsRestClassifier(svr_rbf,-1).fit(train_hist, train_data_number).score(test_hist,test_data_number)
    return accuracy*100

# 1342副图像做训练 789副图像做测试       识别率96.1977186311787%
# 789副图像做训练 1342副图像做测试      识别率96.57228017883756%
if __name__ == '__main__':
    start = time.time()
    print("识别率{}%".format(accuracy_svm()))
    end  = time.time()
    print("程序执行时间是{}".format(end - start)) # 程序执行时间是104.615962266922



























