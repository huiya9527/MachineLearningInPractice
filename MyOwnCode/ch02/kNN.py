# coding:utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

###
# @param inX：输入向量
# @param dataSet: 训练样本集
# @param labels: 标签向量集
# @param k: 最近邻个数

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 样本的个数
    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 分类数据和样本做差值
    sqDiffMat = diffMat**2  # 差值平方
    sqDistances = sqDiffMat.sum(axis=1)  # 最低维度相加
    distances = sqDistances**0.5  # 求和后开方
    sortedDistIndicies = distances.argsort()  # 返回距离从小到大的索引值
    classCount = {}
    # 选择距离最小的K个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 没有值设为0， 累计
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 字典按值倒排
    return sortedClassCount[0][0]

