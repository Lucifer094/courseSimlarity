# -*- coding:utf-8 -*-

############# by k 2020/2/28
############# 12.修正K-Medoids
# 利用模拟退火，给出修正小数
import numpy as np
initT = 1000
minT = 1
iterL = 80
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
k = 150
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
k = 150
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



import csv
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
import pandas as pd
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\centersLables.csv'
centersDF = pd.read_csv(sourcePath, encoding='utf-8', index_col=0, header=None)
szTXcorpus = r"C:\Users\king\Desktop\king\Course\Course\user\my-small.txt"  # 小语料库
wv = KeyedVectors.load_word2vec_format(szTXcorpus, binary=False)
disCenDF = pd.DataFrame(np.zeros((len(centersDF.index), len(centersDF.index))), index=centersDF.index, columns=centersDF.index)
for pos1 in range(len(centersDF.index)):
    word1 = centersDF.index[pos1]
    for pos2 in range(pos1, len(centersDF.index)):
        word2 = centersDF.index[pos2]
        result = wv.similarity(word1, word2)
        disCenDF.iloc[pos1, pos2] = result
        disCenDF.iloc[pos2, pos1] = result
disCenAr = disCenDF.values
# 导入词类别结果
sourcePath = r'C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\wordsLables.csv'
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
            word1 = int(wordList1[i])
            for j in range(len(wordList2)):
                word2 = int(wordList2[j])
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
    storePath = r'C:\Users\king\Desktop\king\Course\Course\data\NKMedoids\similarityCourse.csv'
    courseDisDF.to_csv(storePath, index=True, header=True, encoding='utf-8')

