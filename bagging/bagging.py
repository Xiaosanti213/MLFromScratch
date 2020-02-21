# coding=utf-8
import numpy as np


def getDataSet():
    dataSet = [
        [0.697, 0.460, '是'],
        [0.774, 0.376, '是'],
        [0.634, 0.264, '是'],
        [0.608, 0.318, '是'],
        [0.556, 0.215, '是'],
        [0.403, 0.237, '是'],
        [0.481, 0.149, '是'],
        [0.437, 0.211, '是'],
        [0.666, 0.091, '否'],
        [0.243, 0.267, '否'],
        [0.245, 0.057, '否'],
        [0.343, 0.099, '否'],
        [0.639, 0.161, '否'],
        [0.657, 0.198, '否'],
        [0.360, 0.370, '否'],
        [0.593, 0.042, '否'],
        [0.719, 0.103, '否']
    ]

    for i in range(len(dataSet)):
        if dataSet[i][-1] == '是':
            dataSet[i][-1] = 1
        else:
            dataSet[i][-1] = -1

    return np.array(dataSet)


def bootstrap(dataSet):
    """
    采样得到与源数据集相同大小的数据集。
    :param dataSet:
    :return:
    """
    n = len(dataSet)
    index = np.random.randint(0, n, n)
    newData = dataSet[index]
    return newData


def calErr(dataSet, feature, threshVal, inequal):
    """
    计算决策树桩的错误率
    :param dataSet:     数据集
    :param feature:     属性index
    :param threshVal:   属性阈值
    :param inequal:     不等号
    :return:            错误率
    """
    errCnt = 0
    if inequal == 'lt':
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or \
                    (data[feature] > threshVal and data[-1] == 1):
                errCnt += 1
    else:
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or \
                    (data[feature] < threshVal and data[-1] == 1):
                errCnt += 1
    return errCnt / float(len(dataSet))


def buildStump(dataSet):
    """
    决策树桩
    :param dataSet:
    :return:    返回储存决策树桩的字典。
    """
    m, n = dataSet.shape
    bestErr = np.inf
    bestStump = {}
    for i in range(n - 1):
        for j in range(m):
            threVal = dataSet[j][i]
            for inequal in ['lt', 'gt']:
                err = calErr(dataSet, i, threVal, inequal)
                if err < bestErr:
                    bestErr = err
                    bestStump["feature"] = i
                    bestStump["threshVal"] = threVal
                    bestStump["inequal"] = inequal
                    bestStump["err"] = err

    return bestStump


def predict(data, bestStump):
    """
    通过决策树桩预测数据的类别。
    :param data:        待预测的数据。
    :param bestStump:   决策树桩的字典。
    :return:            预测值。
    """
    if bestStump["inequal"] == 'lt':
        if data[bestStump["feature"]] <= bestStump["threshVal"]:
            return 1
        else:
            return -1
    else:
        if data[bestStump["feature"]] >= bestStump["threshVal"]:
            return 1
        else:
            return -1


def calcAcc(dataSet, G):
    """
    计算分类器的准确度
    :param dataSet:    数据集
    :param G:          通过bagging得到的分类器列表。
    :return:           准确度
    """
    rightCnt = 0
    for data in dataSet:
        preCnt = 0
        for g in G:
            preCnt += predict(data, g)
        if (preCnt > 0 and data[-1] == 1) \
                or (preCnt <= 0 and data[-1] == -1):
            rightCnt += 1
    return rightCnt / float(len(dataSet))


def bagging(dataSet, T):
    """
    通过bootstrap从原数据中得到新的数据，训练新的分类器。
    :param dataSet:     数据集
    :param T:           分类器数量
    :return:            分类器列表
    """
    G = []
    for t in range(T):
        newDataSet = bootstrap(dataSet)
        stump = buildStump(newDataSet)
        G.append(stump)
    return G


def main():
    dataSet = getDataSet()
    for t in [3, 5, 11]:
        G = bagging(dataSet, 11)
        #print(f"G{t} = ", G)
        print("prediction = ", calcAcc(dataSet, G))


if __name__ == '__main__':
    main()
