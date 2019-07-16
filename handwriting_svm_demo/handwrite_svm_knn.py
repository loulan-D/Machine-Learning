import numpy as np
from PIL import Image
import os
import glob
from sklearn import svm
from sklearn import neighbors


def img_to_vector(filename):
    """
    图片数据转化成向量：将50x50的图像数据转换成1x2500的numpy向量
    灰度化 --> 归一化(0-1) --> numpy向量
    :params filename: 图片文件名; 如./Sample1/1_5.png
    :return: 1x2500的numpy向量
    """
    img = Image.open(filename).convert('L')
    img_arr = np.array(img)
    img_normlization = img_arr/255      # 归一化处理，将像素值0-255映射到0-1；
    img_arr = np.reshape(img_normlization,(1, -1))
    return img_arr


def get_data_and_table(data_path):
    """
    得到训练数据或测试数据的数据集和对应的标签；训练数据 | 测试数据 按data_name的名字判断
    :params data_path: 训练数据或测试数据的路径
    :return : 返回训练数据或测试数据的 数据集和标签
    """
    data_name = os.path.split(data_path)[0].split('/')[2]
    if 'train' in data_name:
        mat_shape = (400, 2500)   # 训练数据对应的矩阵shape 
    else:
        mat_shape = (100, 2500)
    child_path = ['Sample5/', 'Sample4/', 'Sample3/', 'Sample2/', 'Sample1/']
    data = []
    table = []
    for path in child_path:
        path = data_path + path + '*.png'
        for image in glob.glob(path):
            data.append(img_to_vector(image))
            table.append(os.path.basename(image)[0])
    data = np.reshape(data, mat_shape)
    table = np.array(table)
    print(f"{data_name} data shape: {data.shape}")
    print(f"{data_name} table shape: {table.shape}")
    return data, table


def svm_model(train_data, train_table, test_data, test_table):
    """
    使用sklearn中的svm函数创建一个SVM model
    :params train_data: 训练数据集
    :params train_table: 训练标签
    :params test_data: 测试数据集
    :params test_table: 测试标签
    :return :
    """
    model = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf',probability=True)
    model = model.fit(train_data, train_table)
    result = model.predict(test_data)
    print(f"测试样本属于某种类别的概率: {model.predict_proba(test_data)}")
    print(f"测试样本到对应超平面的距离: {model.decision_function(test_data)}")
    print(f"预测结果: {result}")
    print(f"真实结果: {test_table}")    
    score = model.score(test_data, test_table)
    print(f"SVM Model Score:{score}")


def knn_model(train_data, train_table, test_data, test_table):
    """
    使用sklearn的neighbors函数创建一个knn model
    :params train_data: 训练数据集
    :params train_table: 训练标签
    :params test_data:  测试数据集
    :params test_table: 测试标签
    :return :
    """
    for n in [1,2,3,4,5,6]:
        for weights in ['uniform', 'distance']:
            model = neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights)
            model.fit(train_data, train_table)
            # result = model.predict(test_data)
            # print(test_table, result)
            score = model.score(test_data, test_table)
            print(f"KNN n_neightbors: {n} weight: {weights} score: {score}")


if __name__ == '__main__':
    train_data_path = './dataset/train_dataset/'
    test_data_path = './dataset/test_dataset/'
    train_data, train_table = get_data_and_table(train_data_path)
    test_data, test_table = get_data_and_table(test_data_path)
    svm_model(train_data, train_table, test_data, test_table)
    knn_model(train_data, train_table, test_data, test_table)