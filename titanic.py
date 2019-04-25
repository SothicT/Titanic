#-*- coding:utf-8 –*-

#数据分析库
import pandas as pd
#科学计算库
import numpy as np
# 线性回归
from sklearn.linear_model import LinearRegression
# K折交叉验证
from sklearn.model_selection import KFold
# 绘制ROC曲线 计算AUC值
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# 绘图
import matplotlib.pyplot as plt
# 保存模型
from sklearn.externals import joblib


# 获取数据集
def dataSet():
    data_train = pd.read_csv("dataset/train.csv")
    data_test = pd.read_csv("dataset/test.csv")
    data_test_target = pd.read_csv("dataset/gender_submission.csv")
    return data_train, data_test, data_test_target

# 数据集预处理
# 可自行改变预处理方法，比如0值填充，前后的平均值等等
def dataPreProcessing(dataset):
    # Age列中的缺失值用Age中位数进行填充
    dataset["Age"] = dataset['Age'].fillna(dataset['Age'].median())
    dataset["Fare"] = dataset['Fare'].fillna(dataset['Fare'].median())


# 选取简单的可用输入特征
# 此处为特征工程，主要可以使用降维的相关操作，例如PCA降维等等，可自行改变选取的特征属性
def featureSelect():
    predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    return predictors

# 模型训练
# 可使用不同的模型进行训练，验证方法也可以是5折，10折等等
def trainModel(data_train):
    # 初始化线性回归算法
    alg = LinearRegression()
    # 样本平均分成3份，3折交叉验证
    kf = KFold(n_splits=3, shuffle=False, random_state=1)

    predictions = []
    # 使用三折交叉验证
    for train, test in kf.split(data_train):
        # 选取训练特征向量
        train_predictors = (data_train[featureSelect()].iloc[train, :])
        # 每个特征向量对应的目标
        train_target = data_train["Survived"].iloc[train]
        # 使用特征向量和对应的存活分类训练模型
        alg.fit(train_predictors, train_target)
        # 使用三折划分的其他数据测试模型预测情况
        test_predictions = alg.predict(data_train[featureSelect()].iloc[test, :])
        predictions.append(test_predictions)
    # 按行拼接三次预测结果
    predictions = np.concatenate(predictions, axis=0)

    # Map predictions to outcomes(only possible outcomes are 1 and 0)
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0
    accuracy = sum(predictions == data_train["Survived"]) * 1.0 / len(predictions)
    print("训练准确率为: ", accuracy)
    return alg

# 使用模型预测乘客存活状态
def modelPrediction(alg, data_test):
    # 提取特征向量
    test_predictors = data_test[featureSelect()]
    # 预测每一个特征向量对应的存活状态
    test_predictions = alg.predict(test_predictors)
    test_predictions[test_predictions > .5] = 1
    test_predictions[test_predictions <= .5] = 0
    return test_predictions

# 绘制ROC曲线和得到AUC值
def getROCAndAUC(predictions, data_test_target):
    data_target = data_test_target.iloc[:, 1]
    print(sum(predictions) * 1.0 / len(data_target))
    # 获取FPR和TPR向量
    fpr, tpr, thresholds = roc_curve(data_target, predictions)
    # 计算AUC值
    auc = roc_auc_score(data_target, predictions)
    # 绘制ROC曲线
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title("titanic Receiver operating characteristic")
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr)
    plt.legend(('roc_auc_score=%.3f'%(auc), 'standard'), loc='lower right')
    plt.show()
    return auc

# 主函数，可以使用main的方式运行代码，也可以在IPython中单步运行每个函数
if __name__ == '__main__':
    # 获取数据集
    data_train, data_test, data_test_target = dataSet()
    # 数据预处理
    dataPreProcessing(data_train)
    dataPreProcessing(data_test)
    # 训练模型
    alg = trainModel(data_train)
    # 保存模型
    joblib.dump(alg, 'titanic_model.m')
    # 测试模型
    predictions = modelPrediction(alg, data_test)
    # 数据分析，得到ROC曲线和AUC值
    auc = getROCAndAUC(predictions, data_test_target)
