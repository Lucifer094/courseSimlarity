#!/usr/bin/env python
# -*- coding: utf-8 -*-
############# 将给定的txt文件进行分词
# cut(pathSource, pathDictionary, cutType, pathStore, storeType, fileFormat)
# input parameter：
#    pathSource: 源文件路径，路径下为utf-8编码的多个txt文件
#    pathDictionary: 自定义词典路径
#    cutType:
#        single-分割结果不同的词只出现一次；
#        all-分割结果完整储存；
#    pathStore: 结果存储路径,one-路径+文件名；separate-路径；
#    storeType:
#        one-存储在一个文件中
#        separate-存储在多个文件中
#    fileFormat:
#        txt-支持storeType中的one和separate,存储格式为utf-8
#        csv-支持storeType中的one,存储格式为utf-8
##############  by k 2019/6/16


import re
import os
import jieba
import pandas as pd


def readTxtFile(filename, ec): # 系统默认gb2312， 大文件常用'UTF-8'
    str=""
    with open(filename, 'r', encoding=ec) as f:  # 设置文件对象
        str = f.read()  # 可以是随便对文件的操作
    return(str)


def buildStopWordList(strStop):
    stopwords = set()
    strSplit = strStop.split('\n')
    for line in strSplit:
        stopwords.add(line.strip())
    stopwords.add('\n')
    stopwords.add('\t')
    stopwords.add(' ')
    return stopwords


def readDir(szDir):
    lstFile = []
    for file in os.listdir(szDir):
        file_path = os.path.join(szDir, file)
        lstFile.append(file_path)
    return lstFile


def buildWordSet(str, setStop):  #根据停用词过滤，并利用set去重
    # 过滤数字以及符号
    reg = "[^A-Za-z\u4e00-\u9fa5]"
    # 将分词、去停用词后的文本数据存储在list类型的texts中
    words = ' '.join(jieba.cut(str)).split(' ')  # 利用jieba工具进行中文分词
    setStr = set()
    # 过滤停用词，只保留不属于停用词的词语
    for word in words:
        word = re.sub(reg, '', word)
        if word is not '':
            if word not in setStop:
                setStr.add(word)
            else:
                continue
        else:
            continue
    return setStr


def buildWordList(str, setStop):  #根据停用词过滤
    reg = "[^A-Za-z\u4e00-\u9fa5]"
    # 将分词、去停用词后的文本数据存储在list类型的texts中
    words = ' '.join(jieba.cut(str)).split(' ')  # 利用jieba工具进行中文分词
    setStr = []
    # 过滤停用词，只保留不属于停用词的词语
    for word in words:
        word = re.sub(reg, '', word)
        if word is not '':
            if word not in setStop:
                setStr.append(word)
            else:
                continue
        else:
            continue
    return setStr


def cut(pathSource, pathDictionary, cutType, pathStore, storeType, fileFormat):
    #################  读取停用词列表
    encoding = 'UTF-8'  # 默认统一编码
    strStop = readTxtFile(pathDictionary, encoding)
    setStop = buildStopWordList(strStop)

    #################  读取一个文件夹中的的各个比较文档
    lstFile = readDir(pathSource)
    # 获取短文件名列表
    lstFileShow = []
    for szFilename in lstFile:
        lstShortFilename = re.search(r'[^\\/:*?"<>|\r\n]+$', szFilename)
        lstMatch = re.findall(r'[^\.]+', lstShortFilename[0])  # 短文件名 a  没有后缀 .b
        lstFileShow.append(lstMatch[0])
    # print(lstFileShow)

    #################  根据分割需求进行分割
    lstDocContent = []
    for szFileName in lstFile:
        str = readTxtFile(szFileName, encoding)  # encoding = 'UTF-8' for large files
        if cutType == 'single':
            str = buildWordSet(str, setStop)
        elif cutType == 'all':
            str = buildWordList(str, setStop)
        lstDocContent.append(str)

    # 根据存储要求进行分别存储
    if storeType == 'one':
        if fileFormat == 'csv':
            # 存储文档分词结果，利用pandas保存问csv文件
            dfWord = pd.DataFrame(data=lstDocContent, index=lstFileShow)
            dfWord.to_csv(pathStore, index=True, header=False, encoding=encoding)
        elif fileFormat == 'txt':
           print('not writing')
    elif storeType == 'separate':
        print('not writing')


