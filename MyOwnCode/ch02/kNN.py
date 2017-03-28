# coding:utf-8
from numpy import *
import numpy
import matplotlib.pyplot as plt
import operator
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

###
# 程序清单2-1 k-近邻算法
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
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 没有值设为0， 累计
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 字典按值倒排
    return sortedClassCount[0][0]


#  程序清单2-2 将文本记录到转换NupPy的解析程序
def file2matrix(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()  # 按行读取文件,一次性读取所有行
    numberOfLines = len(arrayOLines)  # 得到文件行数，填充矩阵
    returnMat = zeros((numberOfLines, 3))  # 3是输入数据维度，可以自行修改
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 程序清单2-3 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet)) 源代码中这行应该可以删除
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))  # 重复创建m列
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print datingDataMat
print datingLabels

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.show()


# 程序清单2-4 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))

# datingClassTest()


# 程序清单2-5 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input('percentage of trime spent palying video games?'))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArray = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArray - minVals) / ranges, normMat, datingLabels, 3)
    print "You will probable like this person: ", resultList[classifierResult - 1]

# classifyPerson()
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# testVector = img2vector('digits/testDigits/0_13.txt')
# print testVector[0, 0:31]


# 程序清单2-6 手写数字识别系统测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print 'the classifier came back with: %d, the real answer is %d' % (classifierResult, classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\bthe total error rate is: %f" % (errorCount/float(mTest))

# handwritingClassTest()







