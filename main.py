# -*- coding:utf-8 -*-
############# 控制程序整体运行流程
# 1.完成文件格式的转换
# 2.提取课程描述信息
# 3.利用jieba分词器，将课程描述文件分词并存储
# 4.获取不同词列表并储存
# 5.利用idf，去除信息量底以及无用的词
# 6.利用tf-idf向量化课程，计算课程间的相似度
# 7.查看所有分词结果是否在字典中，将可用的向量按列存储
# 8.将所有的分词结果进行整理，删去不可用的词
# 9.利用tencent语料库计算所有词之间的相似度，并计算课程间相似度
# 10.使用KMeans聚类词语向量，利用聚类中心间距离代表词相似度，并计算课程间相似度
# 11.使用K-Medoids进行聚类，利用聚类中心间距离代表词相似度，并计算课程间相似度
# 12.修正K-Medoids
############# by k 2019/11/8

import os
CURRENT_PATH = os.path.dirname(__file__)

############# 1.完成文件格式的转换
import format
# 初始化文档格式为txt编码方式为utf-8
source_path = CURRENT_PATH + r'/data/curriculum/pdf'
store_path = CURRENT_PATH + r'/data/curriculum/txt'
format.transform_to_txt(source_path, store_path)

############# 2.提取课程描述信息
import extractCourse
source_path = CURRENT_PATH + r'/data/curriculum/txt'
store_path = CURRENT_PATH + r'/data/course'
extractCourse.extract_course(source_path, store_path)

############# 3.利用jieba分词器，将课程描述文件分词并存储，并将词进行不重复整理并输出
import jiebaCut as jc
sourcePath = CURRENT_PATH + r'/data/course'
dicPath = CURRENT_PATH + r'/user/my中文和符号1960.txt'
storePath = CURRENT_PATH + r'/data/cut/cutResult.csv'
jc.cut(sourcePath, dicPath, 'single', storePath, 'one', 'csv')

############# 4.获取不同词列表并储存，存储结果入csv文件
import csv
sourcePath = CURRENT_PATH + r'/data/cut/cutResult.csv'
storePath = CURRENT_PATH + r'/data/cut/allCutResult.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
allResult = set()   # 利用set实现不重复记录
for line in file:
    flag = True     # 实现文件名称不记录功能
    for word in line:
        if flag:
            flag = False
        else:
            if word != '':
                allResult.add(word)     # 字符不为空是进行记录
            else:
                continue
fileStore = open(storePath, 'w', newline='', encoding='utf-8')
csv.writer(fileStore).writerow(allResult)
fileStore.close()

##################### 5.利用idf，去除信息量底以及无用的词
import csv
sourcePath = CURRENT_PATH + r'/data/cut/allCutResult.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
dict = {}
for line in file:
    for word in line:
        dict[word] = 0
sourcePath = CURRENT_PATH + r'/data/cut/cutResult.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
courseNum = 0
for line in file:
    courseNum = courseNum + 1
    for word in line:
        if word in dict:
            dict[word] = dict[word] + 1
        else:
            continue
storePath = CURRENT_PATH + r'/data/cut/allCutResultIdf.csv'
idfList = []
out = open(storePath, 'a', newline='', encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')
for k in sorted(dict, key=dict.__getitem__, reverse=True):
    temp = [k, dict[k] / courseNum]
    csv_write.writerow(temp)
    if dict[k]/courseNum < 0.618:
        idfList.append(k)
    else:
        continue
storePath = CURRENT_PATH + r'/data/cut/wordsIdf.csv'
out = open(storePath, 'a', newline='', encoding='utf-8')
csv_write = csv.writer(out, dialect='excel')
csv_write.writerow(idfList)

############ 6.利用tf-idf向量化课程，计算课程间的相似度
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import csv
import pandas as pd
sourcePath = CURRENT_PATH + r'/data/cut/cutResultAll.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
corpus = []
courseNameList = []
for line in file:
    flag = True
    temp = ""
    for word in line:
        if flag:
            flag = False
            courseNameList.append(word)
        else:
            if word is not '':
                temp = temp + " " + word
            else:
                continue
    corpus.append(temp)
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
# 存储词
storePath = CURRENT_PATH + r'/data/Tf-Idf/words.csv'
wordDF = pd.DataFrame(data=word)
wordDF.to_csv(storePath, header=0, index=0, encoding='utf-8')
# 存储向量
storePath = CURRENT_PATH + r'/data/Tf-Idf/Tf-IdfVectors.csv'
vectorDF = pd.DataFrame(data=X.toarray())
vectorDF.to_csv(storePath, header=0, index=0, encoding='utf-8')
# 利用Tf-Idf计算课程间的相似度
vectors = X.toarray()
import numpy as np
courseSimDF = np.zeros((len(courseNameList), len(courseNameList)))
for course1 in courseNameList:
    pos1 = courseNameList.index(course1)
    for course2 in courseNameList[pos1:]:
        pos2 = courseNameList.index(course2)
        vec1 = vectors[pos1]
        vec2 = vectors[pos2]
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * (np.linalg.norm(vec2)))
        courseSimDF[pos1][pos2] = similarity
        courseSimDF[pos2][pos1] = similarity
    print(pos1)
# 存储课程名称以及相似度矩阵
storePath = CURRENT_PATH + r'/data/Tf-Idf/similarityCourse.csv'
courseDisDF = pd.DataFrame(data=courseSimDF, index=courseNameList, columns=courseNameList)
courseDisDF.to_csv(storePath, header=True, index=True, encoding='utf-8')


############# 7.查看所有分词结果是否在字典中，将可用的向量按列存储
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
# 导入语料库
# corpusPath = r"D:\GitHub\Course\user\500000-small.txt"
corpusPath = CURRENT_PATH + r"/user/500000-small.txt"
wv = KeyedVectors.load_word2vec_format(corpusPath, binary=False)
# 判断词是否在语料库中并寻找向量
dfIndex = []     # 可以使用的词
vectorWords = []        # 可用词的向量
# sourcePath = r'D:\GitHub\Course\data\cut\wordsIdf.csv'
sourcePath = CURRENT_PATH + r'/data/cut/wordsIdf.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
allWord = []
for line in file:
    for word in line:
        allWord.append(word)
for word in allWord:
    try:
        vector = wv[word]
        dfIndex.append(word)
        vectorWords.append(vector)
    except:
        continue
# 将可用的词按列存储
# storeWords = r'D:\GitHub\Course\data\tencent\dfIndex.csv'
storeWords = CURRENT_PATH + r'/data/tencent/dfIndex.csv'
f_csv = open(storeWords, 'w', newline='', encoding='utf-8')
csv.writer(f_csv).writerow(dfIndex)
f_csv.close()
# 将可用的向量按列存储
# storeVectors = r'D:\GitHub\Course\data\tencent\vectorsUseful.csv'
storeVectors = CURRENT_PATH + r'/data/tencent/vectorsUseful.csv'
f_csv = open(storeVectors, 'a', newline='', encoding='utf-8')
f = csv.writer(f_csv, dialect='excel')
for vector in vectorWords:
    f.writerow(vector)
f_csv.close()

############# 8.将所有的分词结果进行整理，删去不可用的词
import csv
import pandas as pd
sourcePath = r'D:\GitHub\Course\data\tencent\dfIndex.csv'
# sourcePath = CURRENT_PATH + r'/data/matrix/dfIndex.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
wordsUseful = []
for line in file:
    for word in line:
        wordsUseful.append(word)
sourcePath = r'D:\GitHub\Course\data\cut\cutResult.csv'
storePath = r'D:\GitHub\Course\data\tencent\cutResultUseful.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
allCutResultUseful = []
for line in file:
    flag = False
    temp = []
    for word in line:
        if flag:
            if word is not '':
                if word in wordsUseful:
                    temp.append(word)
                else:
                    continue
            else:
                continue
        else:
            temp.append(word)
            flag = True
            # print(word)
    allCutResultUseful.append(temp)
dfWords = pd.DataFrame(data=allCutResultUseful)
dfWords.to_csv(storePath, index=False, header=False, encoding='utf-8')

#################### 9.利用tencent语料库计算所有词之间的相似度
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
# # 读取所有可用词
# sourcePath = r'D:\GitHub\Course\data\tencent\dfIndex.csv'
# file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
# wordsUseful = []
# for line in file:
#     for word in line:
#         wordsUseful.append(word)
# # 创建所有可用词之间的距离矩阵
# wordsAR = np.zeros((len(wordsUseful),len(wordsUseful)))
# for pos1 in range(len(wordsUseful)):
#     for pos2 in range(pos1, len(wordsUseful)):
#         temp = wv.similarity(wordsUseful[pos1], wordsUseful[pos2])
#         wordsAR[pos1][pos2] = temp
#         wordsAR[pos2][pos1] = temp
#     print(pos1)
# 导入腾讯语料库模型
# szTXcorpus = r"D:\GitHub\Course\user\500000-small.txt"  # 小语料库
szTXcorpus = r"D:\zhangxin\Course\user\500000-small.txt"  # 小语料库
wv = KeyedVectors.load_word2vec_format(szTXcorpus, binary=False)
# 读取课程分词结果
# sourcePath = r'D:\GitHub\Course\data\tencent\cutResultUseful.csv'
sourcePath = r'D:\zhangxin\Course\data\tencent\cutResultUseful.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
courseNameList = []
courseWords = []
for line in file:
    flag = False
    temp = []
    for word in line:
        if word is not '':
            if flag:
                temp.append(word)
            else:
                courseNameList.append(word)
                flag = True
        else:
            continue
    courseWords.append(temp)
# 根据课程间词相似度计算课程间相似度方式
def countDis(disDF, type=1):
    if type == 1:
        v0 = np.argmax(disDF, axis=0)
        v0 = disDF[v0, range(disDF.shape[1])]
        v1 = np.argmax(disDF, axis=1)
        v1 = disDF[range(disDF.shape[0]), v1]
        result = ((v0.sum() + v1.sum()) / (len(v0) + len(v1)))
    else:
        result = 0
    return result
# 创建课程相似度矩阵
courseNameComputerList = []
for name in courseNameList:
    if "计算机科学与技术" in name:
        courseNameComputerList.append(name)
    else:
        continue
sourcePath = r'D:\zhangxin\Course\data\tencent\courseDis300-400.csv'
courseSimDF = pd.read_csv(sourcePath, header=0, index_col=0, encoding='utf-8')
# courseSimDF = pd.DataFrame(np.zeros((len(courseNameComputerList), len(courseNameList))), index=courseNameComputerList, columns= courseNameList)
for course1 in courseNameComputerList[314:400]:
    print(course1, courseNameComputerList.index(course1), '300-400')
    coursePos1 = courseNameList.index(course1)
    wordList1 = courseWords[coursePos1]
    words1len = len(wordList1)
    for course2 in courseNameList:
        coursePos2 = courseNameList.index(course2)
        wordList2 = courseWords[coursePos2]
        words2len = len(wordList2)
        # 课程间词之间的相似度矩阵
        wordsSim = np.zeros((words1len, words2len))
        for word1 in wordList1:
            word1Pos = wordList1.index(word1)
            for word2 in wordList2:
                word2Pos = wordList2.index(word2)
                # similarity = wordsAR[wordsUseful.index(word1)][wordsUseful.index(word2)]
                similarity = wv.similarity(word1, word2)
                wordsSim[word1Pos][word2Pos] = similarity
        # 计算课程间相似度
        distance = countDis(wordsSim, type=1)
        courseSimDF.iloc[courseNameComputerList.index(course1)][coursePos2] = distance
    # print(courseSimDF.loc[course1,:])
    storePath = r'D:\zhangxin\Course\data\tencent\courseDis300-400.csv'
    courseSimDF.to_csv(storePath, header=True, index=True, encoding='utf-8')
# 存储课程相似度矩阵
storePath = r'D:\GitHub\Course\data\tencent\courseDis.csv'
courseSimDF.to_csv(storePath, header=True, index=True, encoding='utf-8')

storePath = r'C:\Users\Administrator\Desktop\Course\data\tencent\courseDis.csv'
courseSimDF.to_csv(storePath, header=True, index=True, encoding='utf-8')


#################### 10.使用KMeans聚类词语向量，利用聚类中心间距离代表词距离
import csv
# 归一化向量数据，使得余弦距离和欧氏距离近似相等
import numpy as np
import pandas as pd
dataVectors = r'D:\GitHub\Course\data\tencent\vectorsUseful.csv'
vectorDF = pd.read_csv(dataVectors, header=None, index_col=False)
from sklearn import preprocessing
vectorsArray = vectorDF.values
vectorNorm = preprocessing.normalize(vectorsArray, norm='l2')
vectorNDF = pd.DataFrame(data=vectorNorm)
storePath = r'D:\GitHub\Course\data\tencent\vectorsNorm.csv'
vectorNDF.to_csv(storePath, index=False, header=False)
# 肘形法确定最佳簇数
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
dataVectors = r'D:\GitHub\Course\data\tencent\vectorsNorm.csv'
vectorNDF = pd.read_csv(dataVectors, header=None, index_col=False)
vectorNorm = vectorNDF.values
# 使用肘方法确定簇的最佳数量
distortions = []
for i in range(1, 1500, 10):
    km = KMeans(n_clusters=i, max_iter=100, n_init=5, init='k-means++', n_jobs=-1)
    result = km.fit_predict(vectorNorm)
    distortions.append(km.inertia_)
    print(i, km.inertia_)
plt.plot(range(1, 1500, 10), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# 存储手肘法的数据
clusterPath = r'D:\GitHub\Course\data\Kmeans\distancesCluster.csv'
f_csv = open(clusterPath, 'a', newline='', encoding='utf-8')
f = csv.writer(f_csv, dialect='excel')
f.writerow(distortions)
f_csv.close()
# 最佳簇数下使用K-Means进行聚类，簇数为150
from sklearn.cluster import KMeans
km = KMeans(n_clusters=150, max_iter=30000, n_init=40, init='k-means++', n_jobs=-1)
result = km.fit_predict(vectorNorm)
# 将每个数据属于的类别进行存储，并将每个类别的簇中心进行存储
clusterPath = r'D:\GitHub\Course\data\Kmeans\clusterResult.csv'
f_csv = open(clusterPath, 'a', newline='', encoding='utf-8')
f = csv.writer(f_csv, dialect='excel')
f.writerow(result)
f_csv.close()
clusterCenter = r'D:\GitHub\Course\data\Kmeans\clusterCenter.csv'
f_csv = open(clusterCenter, 'a', newline='', encoding='utf-8')
f = csv.writer(f_csv, dialect='excel')
for row in km.cluster_centers_:
    f.writerow(row)
f_csv.close()
# 使用余弦相似度计算聚类中心间的距离
import numpy as np
import pandas as pd
from pandas import DataFrame
sourcePath = r'D:\GitHub\Course\data\Kmeans\clusterCenter.csv'
centerVector = np.loadtxt(sourcePath, delimiter=",")
centerNumber = centerVector.shape[0]
# 聚类中心间的距离矩阵
similarityCenters = DataFrame(np.zeros((centerNumber, centerNumber)))
for i in range(centerNumber):
    vector1 = centerVector[i]
    for j in range(i, centerNumber):
        vector2 = centerVector[j]
        result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        similarityCenters.loc[i, j] = result
        similarityCenters.loc[j, i] = result
storePath = r'D:\GitHub\Course\data\Kmeans\similarityBetweenCenters.csv'
similarityCenters.to_csv(storePath, index=True, header=True, encoding='utf-8')
# 根据聚类中心间的距离，计算课程间的距离
import pandas as pd
import numpy as np
# 读取簇中心距离矩阵
sourcePath = r'D:\GitHub\Course\data\Kmeans\similarityBetweenCenters.csv'
distanceCentersDF = pd.read_csv(sourcePath, header=0, index_col=0, encoding='utf-8')
distanceCenters = distanceCentersDF.values
# 读取词列表以及对应的聚类中心，建立DF方便计算
sourcePath = r'D:\GitHub\Course\data\tencent\dfIndex.csv'
wordList = []
file = csv.reader(open(sourcePath, encoding='utf-8'))
for line in file:
    for word in line:
        wordList.append(word)
sourcePath = r'D:\GitHub\Course\data\Kmeans\clusterResult.csv'
wordCluster = pd.read_csv(sourcePath, names=wordList, encoding='utf-8')
# 将文本中的词转化为相应的聚类结果并储存，从而方便计算
sourcePath = r'D:\GitHub\Course\data\tencent\cutResultUseful.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
courseNameList = []
wordList = []
for line in file:
    flag = True  # 记录文件名称
    temp = []
    for word in line:
        if flag:
            courseNameList.append(word)
            flag = False
        else:
            if word != '':
                temp.append(wordCluster.ix[0, word])  # 字符不为空是进行记录
            else:
                continue
    wordList.append(temp)
# wordDF = pd.DataFrame(data=wordList, index=courseNameList)
# # storePath = r'D:\GitHub\courseAnalyse\data\kMeans\cutWordCenter.csv'
# storePath = CURRENT_PATH + r'/data/kMeans/cutWordCenter.csv'
# wordDF.to_csv(storePath, index=True, header=False, encoding='utf-8')
# 生成课程间距离矩阵
courseNameComputerList = []
for name in courseNameList:
    if "计算机科学与技术" in name:
        courseNameComputerList.append(name)
    else:
        continue
courseDisDF = pd.DataFrame(np.zeros((len(courseNameComputerList), len(courseNameList))), index=courseNameComputerList, columns=courseNameList)
# 根据课程间词距离计算课程间距离方式
def countDis(disDF, type=1):
    if type == 1:
        v0 = np.argmax(disDF, axis=0)
        v0 = disDF[v0, range(disDF.shape[1])]
        v1 = np.argmax(disDF, axis=1)
        v1 = disDF[range(disDF.shape[0]), v1]
        result = ((v0.sum() + v1.sum()) / (len(v0) + len(v1)))
    else:
        result = 0
    return result
# 生成课程间词距离矩阵
for course1 in courseNameComputerList:
    print("course1:", course1, courseNameComputerList.index(course1))
    for course2 in courseNameList:
        print("course2:", course2, courseNameList.index(course1), courseNameList.index(course2))
        wordList1 = wordList[courseNameList.index(course1)]
        wordList2 = wordList[courseNameList.index(course2)]
        lenCourse1 = len(wordList1)
        lenCourse2 = len(wordList2)
        wordDisDF = np.zeros((lenCourse1, lenCourse2))
        for i in range(len(wordList1)):
            word1 = wordList1[i]
            for j in range(len(wordList2)):
                word2 = wordList2[j]
                if wordDisDF[i, j] == 0:
                    try:
                        temp = distanceCenters[word1, word2]
                        wordDisDF[i, j] = temp
                    except:
                        continue
                else:
                    continue
        temp = countDis(wordDisDF, type=1)
        courseDisDF.loc[course1, course2] = temp
        print(temp)
# 保存课程间距离文件
storePath = r'D:\GitHub\Course\data\Kmeans\similarityCourse.csv'
courseDisDF.to_csv(storePath, index=True, header=True, encoding='utf-8')


############# 11.使用K-Medoids进行聚类
from pyclust import KMedoids
import csv
import pandas as pd
# 导入所有单词
sourcePath = r'D:\GitHub\Course\data\tencent\dfIndex.csv'
wordsAllDF = pd.read_csv(sourcePath, header=None, index_col=False, encoding='utf-8')
wordsAllLst = []
for i in wordsAllDF.columns:
    wordsAllLst.append(wordsAllDF.iloc[0, i])
# 导入单词向量
sourcePath = r'D:\GitHub\Course\data\tencent\vectorsNorm.csv'
vectorsDF = pd.read_csv(sourcePath, header=None, index_col=False, encoding='utf-8')
vectorsDF.index = wordsAllLst
# 开始K-Medoids聚类
KMD = KMedoids(n_clusters=500, n_trials=50, distance='euclidean', max_iter=500)
KMD.fit(vectorsDF.values)
# labels_   :  cluster labels for each data item
# centers_  :  cluster centers
# sse_arr_  :  array of SSE values for each cluster
# n_iter_   :  number of iterations for the best trial
# 类别将结果保存
labelsDF = pd.DataFrame(data=KMD.labels_, index=wordsAllLst)
storePath = r'D:\GitHub\Course\data\KMedoids\labels.csv'
labelsDF.to_csv(storePath, encoding='utf-8', index=True, header=False)
# 将类别中心单词和向量同时储存
centersDF = pd.DataFrame(data=KMD.centers_)
centersWordsLst = []
for i in centersDF.index:
    print(i)
    for word in wordsAllLst:
        if (centersDF.iloc[i, :] == vectorsDF.loc[word, :]).all():
            centersWordsLst.append(word)
            print(word)
            break
centersDF.index = centersWordsLst
storePath = r'D:\GitHub\Course\data\KMedoids\centers.csv'
centersDF.to_csv(storePath, encoding='utf-8', index=True, header=False)
# 将类别中心对应的sse分别储存
sseDF = pd.DataFrame(data=KMD.sse_arr_, index=centersWordsLst)
storePath = r'D:\GitHub\Course\data\KMedoids\sse.csv'
sseDF.to_csv(storePath, encoding='utf-8', index=True, header=False)
# 将聚类中心所对应的单词进行整理
wordsSort = []
for word in centersWordsLst:
    temp = []
    temp.append(word)
    wordsSort.append(temp)
for word in labelsDF.index:
    wordsSort[labelsDF.loc[word, 0]].append(word)
wordsSortDF = pd.DataFrame(data=wordsSort)
storePath = r'D:\GitHub\Course\data\KMedoids\centersWords.csv'
wordsSortDF.to_csv(storePath, encoding='utf-8', index=False, header=False)
# 主题聚类结果中心间距离计算
import pandas as pd
sourcePath = r'D:\GitHub\Course\data\KMedoids\centers500.csv'
centersDF = pd.read_csv(sourcePath, encoding='utf-8', index_col=0, header=None)
import numpy as np
disCenDF = pd.DataFrame(np.zeros((len(centersDF.index),len(centersDF.index))),index=centersDF.index,columns=centersDF.index)
for pos1 in range(len(centersDF.index)):
    vector1 = centersDF.iloc[pos1,:]
    for pos2 in range(pos1,len(centersDF.index)):
        vector2=centersDF.iloc[pos2,:]
        result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        disCenDF.iloc[pos1,pos2]=result
        disCenDF.iloc[pos2,pos1]=result
disCenAr = disCenDF.values
# 导入词类别结果
sourcePath = r'D:\GitHub\Course\data\KMedoids\labels500.csv'
wordCluster = pd.read_csv(sourcePath, header=None, index_col=0, encoding='utf-8')
# 导入课程分词结果
import csv
sourcePath = r'D:\GitHub\Course\data\tencent\cutResultUseful.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
courseNameList = []
wordList = []
for line in file:
    flag = True  # 记录文件名称
    temp = []
    for word in line:
        if flag:
            courseNameList.append(word)
            flag = False
        else:
            if word != '':
                temp.append(wordCluster.loc[word,1])  # 字符不为空是进行记录
            else:
                continue
    wordList.append(temp)
# 课程间距离计算
courseNameComputerList = []
for name in courseNameList:
    if "计算机科学与技术" in name:
        courseNameComputerList.append(name)
    else:
        continue
courseDisDF = pd.DataFrame(np.zeros((len(courseNameComputerList), len(courseNameList))), index=courseNameComputerList, columns=courseNameList)
def countDis(disDF, type=1):
    if type == 1:
        v0 = np.argmax(disDF, axis=0)
        v0 = disDF[v0, range(disDF.shape[1])]
        v1 = np.argmax(disDF, axis=1)
        v1 = disDF[range(disDF.shape[0]), v1]
        result = ((v0.sum() + v1.sum()) / (len(v0) + len(v1)))
    else:
        result = 0
    return result
for course1 in courseNameComputerList:
    print("course1:", course1, courseNameComputerList.index(course1), "K=150")
    for course2 in courseNameList:
        wordList1 = wordList[courseNameList.index(course1)]
        wordList2 = wordList[courseNameList.index(course2)]
        lenCourse1 = len(wordList1)
        lenCourse2 = len(wordList2)
        wordDisAR = np.zeros((lenCourse1, lenCourse2))
        for i in range(len(wordList1)):
            word1 = wordList1[i]
            for j in range(len(wordList2)):
                word2 = wordList2[j]
                if wordDisAR[i, j] == 0:
                    try:
                        temp = disCenAr[word1, word2]
                        wordDisAR[i, j] = temp
                    except:
                        continue
                else:
                    continue
        result = countDis(wordDisAR, type=1)
        courseDisDF.loc[course1, course2] = result
    print(courseDisDF.loc[course1, :])
storePath = r'D:\GitHub\Course\data\KMedoids\similarityCourse.csv'
courseDisDF.to_csv(storePath, index=True, header=True, encoding='utf-8')


############# 12.修正K-Medoids
# 利用模拟退火，给出修正小数
import numpy as np
initT = 1000
minT = 1
iterL = 280
eta = 0.95
kT = 1
nowL = iterL
nowT = initT
def simulatedAnnealing(detaE):
    global nowT
    global nowL
    value = np.exp(-detaE / (kT * nowT))
    if nowL < 0:
        nowL = iterL
        if nowT >= minT:
            nowT = eta * nowT
    else:
        nowL = nowL - 1
    return value

# 每一个词将与其最相似的n/2k个词求平均相似度，并将其数据与相对应的词进行存储，根据大小进行排序，数值最大的前K个词作为初始中心词
import csv
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
sourcePath = r'D:\GitHub\Course\data\tencent\dfIndex.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
wordsUseful = []
for line in file:
    for word in line:
        wordsUseful.append(word)
szTXcorpus = r"C:\Users\king\Desktop\king\Course\Course\user\my-small.txt"  # 小语料库
wv = KeyedVectors.load_word2vec_format(szTXcorpus, binary=False)
k = 500
wordsDic = {}
for word in wordsUseful:
    mostSim = wv.similar_by_word(word, topn=int(len(wordsUseful)/(2*k)))
    value = 0
    for temp in mostSim:
        value = value + temp[1]
    value = value/int(len(wordsUseful)/(2*k))
    wordsDic[word] = value
wordsDicSor = sorted(wordsDic.items(), key=lambda wordsDic:wordsDic[1], reverse=True)
wordsSimDF = pd.DataFrame(wordsDicSor)
storePath = r"C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\wordsSimSor.csv"
wordsSimDF.to_csv(storePath, encoding='utf-8', index=False, header=False)
wordsNewDic = {}
pos = 1
for item in wordsDicSor:
    global pos
    correct = simulatedAnnealing(2*pos/len(wordsUseful))
    word = item[0]
    mostSim = wv.similar_by_word(word, topn=int((len(wordsUseful)/(2*k))*correct))
    value = 0
    for temp in mostSim:
        value = value + temp[1]
    value = value / int((len(wordsUseful)/(2*k))*correct)
    wordsNewDic[word] = value
    pos = pos + 1
wordsNewDicSor = sorted(wordsNewDic.items(), key=lambda wordsNewDic:wordsNewDic[1], reverse=True)
wordsNewSimDF = pd.DataFrame(wordsNewDicSor)
storePath = r"C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\wordsNewSimSor.csv"
wordsNewSimDF.to_csv(storePath, encoding='utf-8', index=False, header=False)

# 在所有点中，从大到小依次进行操作：
# 1.将词典首个放入中心点集中，并从所有点中删除该点；
# 2.从所有点中删除与该点相近的n/2k个词；
# 3.直到中心点够k个为止
szTXcorpus = r"C:\Users\king\Desktop\king\Course\Course\user\my-small.txt"  # 小语料库
wv = KeyedVectors.load_word2vec_format(szTXcorpus, binary=False)
k = 500
sourcePath = r"C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\wordsNewSimSor.csv"
wordsSimDF = pd.read_csv(sourcePath, encoding='utf-8', header=None, index_col=False)
wordsDicSor = wordsSimDF.set_index(0).T.to_dict('list')
wordsCenLst = []
for i in range(k):
    word = list(wordsDicSor.keys())[0]
    del wordsDicSor[word]
    mostSim = wv.similar_by_word(word, topn=int(len(wordsSimDF.index) / (2*k)))
    for temp in mostSim:
        if temp[0] in wordsDicSor:
            del wordsDicSor[temp[0]]
    wordsCenLst.append(word)
centersDF = pd.DataFrame(data=np.zeros((len(wordsCenLst), 0)), index=wordsCenLst)
for word in wordsCenLst:
    centersDF.loc[word, 0] = wordsUseful.index(word)
storePath = r'C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\centersLables.csv'
centersDF.to_csv(storePath, encoding='utf-8', index=True, header=False)
wordsLablesDF = pd.DataFrame(data=np.zeros((len(wordsUseful), 0)), index=wordsUseful)
for word in wordsUseful:
    simValue = 0
    centerWords = wordsCenLst[0]
    for center in wordsCenLst:
        temp = wv.similarity(word, center)
        if temp > simValue:
            simValue = temp
            centerWords = center
    wordsLablesDF.loc[word, 0] = wordsCenLst.index(centerWords)
storePath = r'C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\wordsLables.csv'
wordsLablesDF.to_csv(storePath, encoding='utf-8', index=True, header=False)
# S_Dbw()系数评价聚类结果好坏
import numpy as np
import math
class S_Dbw():
    def __init__(self, data, data_cluster, centers_index):
        """
        data --> raw data
        data_cluster --> The category to which each point belongs(category starts at 0)
        centers_index --> Cluster center index
        """
        self.data = data
        self.data_cluster = data_cluster
        self.centers_index = centers_index
        self.k = len(centers_index)
        # Initialize stdev
        self.stdev = 0
        for i in range(self.k):
            std_matrix_i = np.std(data[self.data_cluster == i], axis=0)
            self.stdev += math.sqrt(np.dot(std_matrix_i.T, std_matrix_i))
        self.stdev = math.sqrt(self.stdev) / self.k
    def density(self, density_list=[]):
        """
        compute the density of one or two cluster(depend on density_list)
        """
        density = 0
        centers_index1 = self.centers_index[density_list[0]]
        if len(density_list) == 2:
            centers_index2 = self.centers_index[density_list[1]]
            center_v = (self.data[centers_index1] + self.data[centers_index2]) / 2
        else:
            center_v = self.data[centers_index1]
        for i in density_list:
            temp = self.data[self.data_cluster == i]
            for j in temp:
                if np.linalg.norm(j - center_v) <= self.stdev:
                    density += 1
        return density
    def Dens_bw(self):
        density_list = []
        result = 0
        for i in range(self.k):
            density_list.append(self.density(density_list=[i]))
        for i in range(self.k):
            for j in range(self.k):
                if i == j:
                    continue
                result += self.density([i, j]) / max(density_list[i], density_list[j])
        return result / (self.k * (self.k - 1))
    def Scat(self):
        theta_s = np.std(self.data, axis=0)
        theta_s_2norm = math.sqrt(np.dot(theta_s.T, theta_s))
        sum_theta_2norm = 0
        for i in range(self.k):
            matrix_data_i = self.data[self.data_cluster == i]
            theta_i = np.std(matrix_data_i, axis=0)
            sum_theta_2norm += math.sqrt(np.dot(theta_i.T, theta_i))
        return sum_theta_2norm / (theta_s_2norm * self.k)
    def S_Dbw_result(self):
        """
        compute the final result
        """
        return self.Dens_bw() + self.Scat()
# 和KMD对比结果
import pandas as pd
import csv
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\tencent\vectorsNorm.csv'
wordsVectorsDF = pd.read_csv(sourcePath, encoding='utf-8', header=None, index_col=False)
wordsVectorsArray = wordsVectorsDF.values
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\KMedoids\centers500.csv'
centersVectorsDF = pd.read_csv(sourcePath, encoding='utf-8', header=None, index_col=0)
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\KMedoids\labels500.csv'
wordsLablesDF = pd.read_csv(sourcePath, encoding='utf-8', header=None, index_col=0)
wordsLables = []
for word in wordsLablesDF.index:
    wordsLables.append(wordsLablesDF.loc[word, 1])
wordsLablesArray = np.array(wordsLables)
sourcePath = r'D:\GitHub\Course\data\tencent\dfIndex.csv'
file = csv.reader(open(sourcePath, 'r', encoding='utf-8'))
wordsUseful = []
for line in file:
    for word in line:
        wordsUseful.append(word)
centersPos = []
for word in centersVectorsDF.index:
    centersPos.append(wordsUseful.index(word))
centersPosArray = np.array(centersPos)
a = S_Dbw(wordsVectorsArray, wordsLablesArray, centersPosArray)
print(a.S_Dbw_result())
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\tencent\vectorsNorm.csv'
wordsVectorsDF = pd.read_csv(sourcePath, encoding='utf-8', header=None, index_col=False)
wordsVectorsArray = wordsVectorsDF.values
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\centersLables.csv'
centersPosDF = pd.read_csv(sourcePath, encoding='utf-8', header=None, index_col=0)
centersPosArray = centersPosDF.values
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\wordsLables.csv'
wordsLablesDF = pd.read_csv(sourcePath, encoding='utf-8', header=None, index_col=0)
wordsLablesArray = wordsLablesDF.values
a = S_Dbw(wordsVectorsArray, wordsLablesArray, centersPosArray)
print(a.S_Dbw_result())


