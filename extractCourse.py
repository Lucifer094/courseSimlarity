# -*- coding:utf-8 -*-
############# 课程信息提取
# 根据不同的课程大纲文件，提取大纲文件中的课程描述信息
# 完成了
# 输入：
#   extract_course(sourcePath, store_path)
# 参数：
#   sourcePath：源文件所在目录
#   store_path：存储路径所在目录
# 输出：
#   课程描述信息提取完成后，每个课程单独存储一个文件
############# by k 2019/11/4
import re


# 根据所给的正则表达式，提取每一门课程信息
def extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end):
    file = open(extract_file_path, 'r', encoding='utf-8')
    txts = file.readlines()
    # 课程信息正在录入标记
    write_is = False
    file_open_is = False
    # 处理教学大纲，提取每一门课程信息
    for txt in txts:
        # 记录是否为课程名称字段、开始标记、结束标记
        title_is = re.search(course_name, txt)
        head_is = re.search(extract_head, txt)
        end_is = re.search(extract_end, txt)
        # 如果是课程标题且无文件进行写入，则新建并打开课程文件，名称为‘学校 课程名称.txt’
        if title_is:
            title = title_is.group()
            # 删去课程名称行各种可能存在的无意义符号
            subject = re.sub('^.*：', '', title)
            subject = re.sub('《', '', subject)
            subject = re.sub("》.*$", '', subject)
            subject = re.sub(' *“', '', subject)
            subject = re.sub("”.*$", '', subject)
            subject = re.sub('/', '_', subject)
            subject = re.sub('课程教学大纲', '', subject)
            store_file_path = store_path + r'/' + university_name + r' ' + subject + '.txt'
            if file_open_is:
                store_file.close()
            store_file = open(store_file_path, 'w', encoding='utf-8')
            file_open_is = True
        # 为开始标记则标记开始写入，为结束标记则标记写入停止并关闭文件
        elif head_is:
            if file_open_is:
                write_is = True
            else:
                continue
        elif end_is:
            if file_open_is:
                store_file.close()
                file_open_is = False
                write_is = False
            else:
                continue
        elif write_is:
            if file_open_is:
                txt = txt.strip()
                # txt = re.sub('PDF 文件使用 "pdfFactory Pro" 试用版本创建 www.fineprint.cn', '', txt)
                store_file.write(txt)
            else:
                continue
        else:
            continue
    file.close()
    print(university_name + "课程信息提取完成！")


def extract_course(source_path, store_path):
    # 根据每一门课程设置不同的课程名称、课程描述的正则表达式进行提取
    # extract_file_path = sourcePath + r'/安徽大学 计算机科学与技术.txt'
    # course_name = r'《.+》教学大纲'
    # extract_head = r'(课程性质与教学目标|课程的教学目标与任务)'
    # extract_end = r'(教学方法|教学安排及方式)'
    # # extract_end = r'(教学内容及基本要求|课程具体内容及基本要求|实验课具体内容及基本要求)'
    # university_name = '安徽大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/中国科学与技术大学 计算机科学与技术.txt'
    # course_name = r'课程名称（中文）：.+'
    # extract_head = r'^主要内容：'
    # extract_end = r'^<返回指导性学习计划表>'
    # university_name = '中国科学与技术大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/西南大学 计算机科学与技术.txt'
    # course_name = r'《.+》本科课程教学大纲$'
    # extract_head = r'课程简介'
    # # extract_head = r'课程教学内容、要求、重难点及学时安排'
    # extract_end = r'考核方式及成绩评定'
    # university_name = '西南大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/潍坊学院 计算机科学与技术.txt'
    # course_name = r'《.+》课程教学大纲$|《.+》教学大纲$'
    # extract_head = r'教学目的和任务'
    # extract_end = r'推荐教材及参考书目'
    # university_name = '潍坊学院 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/沈阳工业大学 计算机科学与技术.txt'
    # course_name = r'《.+》课程教学大纲$'
    # extract_head = r'本课程的性质与任务|课程性质和任务|课程性质与任务|课程性质、目的和任务'
    # # extract_head = r'课程教学目标'
    # extract_end = r'五、其他教学环节|五、教学方法'
    # # extract_end = r'教学内容、基本要求与学时分配'
    # university_name = '沈阳工业大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/曲阜师范大学 计算机科学与技术.txt'
    # course_name = r'“.+”课程教学大纲$'
    # extract_head = r'课程任务目标'
    # extract_end = r'学时分配'
    # university_name = '曲阜师范大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/湖北理工学院 计算机科学与技术.txt'
    # course_name = r'《.+》理论教学大纲$|《.+》实验教学大纲|《.+》教学大纲'
    # extract_head = r'教学内容、基本要求及学时安排|课程设计教学内容|实验教学目的和任务|实验教学的目的'
    # extract_end = r'实践性教学环节|实验考核办法与成绩评定'
    # university_name = '湖北理工学院 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/广州大学 计算机科学与技术.txt'
    # course_name = r'《.+》课程教学大纲$'
    # extract_head = r'教学大纲说明'
    # extract_end = r'学时分配'
    # university_name = '广州大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/北京邮电大学.txt'
    # course_name = r'《.+》课程教学大纲$'
    # extract_head = r'课程教学目的'
    # extract_end = r'课程资源'
    # university_name = '北京邮电大学'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/哈尔滨工程大学 计算机科学与技术.txt'
    # course_name = r'.+课程教学大纲'
    # extract_head = r'课程目.'
    # extract_end = r'参考教材及学习资源'
    # # extract_end = r'教学内容与学时分配'
    # university_name = '哈尔滨工程大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/上海师范大学 计算机科学与技术.txt'
    # course_name = r'《.+》课程教学大纲$'
    # extract_head = r'课程简介'
    # extract_end = r'修读要求'
    # # extract_end = r'教学内容与进度安排'
    # university_name = '上海师范大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    # extract_file_path = sourcePath + r'/西南石油大学 计算机科学与技术.txt'
    # course_name = r'《.+》课程教学大纲$'
    # extract_head = r'目的与任务及能力培养|课程性质和定位'
    # # extract_head = r'目的与任务及能力培养'
    # extract_end = r'考核方式与评分标准|实施建议'
    # # extract_end = r'教学内容、要求及学时分配'
    # university_name = '西南石油大学 计算机科学与技术'
    # extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)
    extract_file_path = source_path + r'/广东海洋大学 计算机科学与技术.txt'
    course_name = r'《.+》课程教学大纲'
    extract_head = r'课程简介'
    extract_end = r'课程考核及成绩评定要求'
    university_name = '广东海洋大学 计算机科学与技术'
    extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)


def extract_course_HY(source_path, store_path):
    import os
    file_list = os.listdir(source_path)
    for file_name in file_list:
        extract_file_path = source_path + r'/' + file_name
        course_name = r'《.+》课程教学大纲|《.+》实验教学大纲'
        extract_head = r'课程简介|课程概述'
        extract_end = r'课程考核及成绩评定要求|课程考核'
        university_name = re.sub('.txt', '', file_name)
        extract(extract_file_path, store_path, university_name, course_name, extract_head, extract_end)


def extractP(extract_file_path, store_path, tag):
    file = open(extract_file_path, 'r', encoding='utf-8')
    txts = file.readlines()
    txtBef = ''
    for txt in txts:
        title_is = re.search(tag, txt)
        if title_is:
            # 截取名称
            courseName = txtBef
            courseName = re.sub('\n', '', courseName)
            # print(courseName)
            # 截取核心以及学时
            describe = re.sub('.*?［', '', txt)
            describe = re.sub('］.*?', '', describe)
            describe = re.sub('\n', '', describe)
            # print(describe)
            # 分情况截取核心描述信息
            if "核心一级" in describe and "核心二级" in describe:
                type = 1
                # 计算核心学时
                num1 = re.sub('个核心一级.*', '', describe)
                num2 = re.sub('核心二级.*', '', describe)
                num2 = re.sub('.*个核心一级', '', num2)
                num2 = re.findall('\d*个', num2)
                num2 = re.sub('个', '', num2[0])
                # print("type1")
                # print(num1)
                # print(num2)
                # 提取核心二级描述信息
                store_path_txt = store_path + r'/' + courseName + ' ' + "核心二级" + ' ' + num2 + '.txt'
                store_file = open(store_path_txt, 'w', encoding='utf-8')
                store_file.close()
                # 提取核心一级描述信息
                store_path_txt = store_path+r'/'+courseName+' '+"核心一级"+ ' '+num1+'.txt'
                store_file = open(store_path_txt, 'w', encoding='utf-8')
                store_file.close()
            elif "核心一级" in describe:
                num1 = re.sub('个核心.*', '', describe)
                store_path_txt = store_path + r'/' + courseName + ' ' + "核心一级" + ' ' + num1 + '.txt'
                store_file = open(store_path_txt, 'w', encoding='utf-8')
                store_file.close()
                # print("type2")
            else:
                num2 = re.sub('个核心.*', '', describe)
                store_path_txt = store_path + r'/' + courseName + ' ' + "核心二级" + ' ' + num2 + '.txt'
                store_file = open(store_path_txt, 'w', encoding='utf-8')
                store_file.close()
                # print("type3")
        txtBef = txt



def extract_point(source_path, store_path):
    import os
    file_list = os.listdir(source_path)
    for file_name in file_list:
        extract_file_path = source_path + r'/' + file_name
        tag = "［.*?个核心.级"
        extractP(extract_file_path, store_path, tag)

