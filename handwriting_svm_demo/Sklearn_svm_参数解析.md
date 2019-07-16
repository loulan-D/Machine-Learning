### 参考文章
- [官网sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

-----
### svm函数介绍
sklearn中的svm源于libsvm
```
from sklearn import svm

svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```
---

### 参数介绍
- C: SVC的惩罚参数，默认值是1.0，C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率搞，但泛化能力弱；C越小，误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
- kernel: 核函数，默认是rbf，可以是'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
- degree: 多项式poly函数的维度，默认是3，选择其他核函数时会被忽略
- gamma: ’rbf‘, 'ploy' 和'sigmoid'的核函数参数。默认是’auto‘,如果是auto,则值为1/n_features
- coef0: 核函数的常数项.对于'poly'和'sigmoid'有用
- probability: 是否采用概率估计.默认为False
- shrinking ：是否采用shrinking heuristic方法，默认为true
- tol ：停止训练的误差值大小，默认为1e-3
- cache_size ：核函数cache缓存大小，默认为200
- class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
- verbose ：是否允许详细输出
- max_iter ：最大迭代次数。-1为无限制。
- decision_function_shape ：‘ovo’, ‘ovr’ or None, default=ovr
- random_state ：数据洗牌时的种子值，int值

-----
### 参数详解

#### kernel:核函数
- **precomputed**: 自己提前计算好核函数矩阵,算法内部不再用核函数去计算核矩阵，而是直接用给的核矩阵.
- **linear**: 线性核函数,主要用于线性可分问题，对于线性可分数据，分类效果很明显
- **poly**: 多项式核函数可以实现将低维的输入空间映射到高维的特征空间，但是多项式核函数的参数多，当多项式的阶数比较高的时候，核矩阵的元素值将趋于无穷大或者无穷小，计算复杂度会大到无法计算。
- **rbf**: 高斯核函数，高斯径向基函数是一种局部性强的核函数，其可以将一个样本映射到一个更高维的空间内，该核函数是应用最广的一个，无论大样本还是小样本都有比较好的性能，而且其相对于多项式核函数参数要少，因此大多数情况下在不知道用什么核函数的时候，优先使用高斯核函数。
- **sigmoid**: sigmoid核函数,采用sigmoid核函数，支持向量机实现的就是一种多层神经网络。 因此，在选用核函数的时候，如果我们对我们的数据有一定的先验知识，就利用先验来选择符合数据分布的核函数；如果不知道的话，通常使用交叉验证的方法，来试用不同的核函数，误差最小的即为效果最好的核函数，或者也可以将多个核函数结合起来。

**核函数的选择**：
- 如果特征的数量大到和样本数量差不多，则选用LR或者线性核的SVM；
- 如果特征的数量小，样本的数量正常，则选用SVM+高斯核函数；
- 如果特征的数量小，而样本的数量很大，则需要手工添加一些特征从而变成第一种情况。

另一种标准：
- 当样本线性可分时，一般选择linear 线性核函数
- 当样本线性不可分时，有很多选择，这里选择rbf 即径向基函数，又称高斯核函数。

#### C：惩罚参数
在泛化能力和准确度间做取舍，一般来说不需要做修改，如果不需要强大的泛化能力，可减小C的值，即：C值越大，在测试集效果越好，但可能过拟合，C值越小，容忍错误的能力越强，在测试集上的效果越差。参数C和gamma是svm中两个非常重要的参数，对其值的调整直接决定了整个模型最终的好坏。
C 寻找 margin 最大的超平面和保证数据点偏差量最小之间的权重。C越大，模型允许的偏差越小。 
松弛变量的系数，称为惩罚系数,用来调整容忍松弛度,当C越大，说明该模型对分类错误更加容忍，也就是为了避免过拟合。

#### gamma
浮点数，作为三个核函数的参数，隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少。gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
支持向量机的间隔，既是超平面距离不同类别的最小距离，是一个float类型的值，可以自己规定，也可以用SVM自己的值，有两个选择
- auto 选择auto时，gamma = 1/feature_num ,也就是特征的数目分之1
- scale 选择scale时，gamma = 1/(feature_num * X.std()), 特征数目乘样本标准差分之1. 一般来说，scael比auto结果准确。

#### C和gamma
相同的C，gamma越大，分类边界离样本越近。相同的gamma，C越大，分类越严格。 

#### decision_function_shape
决策函数类型，在不同版本默认值不一样。如果最后预测函数使用clf.decision_function可以看到效果，输出结果为该类到对应超平面的距离。
- ovr(one versus rest):一对多 训练时依次把某个类别的样本归为一类,其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。
- ovo(one versus one): 一对一 其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM。当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。



#### class_weight: 
字典类型或者‘balance’字符串,默认为None。 给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面指出的参数C.

#### max_iter
int参数 默认为-1，最大迭代次数，如果为-1，表示不限制


----
### SVM方法和属性

#### 方法
- fit: 用于训练SVM，具体参数已经在定义SVM对象的时候给出了，这时候只需要给出数据集X和X对应的标签Y即可
- predict: 基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示每个测试样本的类别
- predict_proba: 基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示测试样本属于每种类型的概率
- decision_function: 基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示测试样本到对应类型的超平面距离
- set_params: 设置SVC函数的参数
- get_params: 获取当前svm函数的各项参数值
- score: 获取预测结果准确率

#### 属性
- svc.nsupport: 各类各有多少个支持向量
- svc.support_: 各类的支持向量在训练样本中的索引
- svc.supportvectors: 各类所有的支持向量
- intercept_: 决策函数中的常量值
- coef_: 特征权重，仅适用于linear内核，来源于dual_coef_和support_vectors_属性
- dualcoef: 决策函数中支持向量的系数值