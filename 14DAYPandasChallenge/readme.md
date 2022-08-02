前言
哈喽大家好，我是不卖焦虑，只聊干货的鲸鲸！
很多小伙伴问我，是不是会sql、excel还有可视化工具（Tableau、powerBI）就可以成为一个数据分析师了，我觉得其实可以，但是得看你想成为一个什么样的数据分析师。
想找个工作不是很难，想找个你满意的数据分析工作还是挺难的。
[图片]
目前Python数据分析适用于众多行业，包括并不局限于网站运营、销售竞争、新媒体传播、互联网公司对数据的分析等。Python数据分析人员主要担任的岗位：企业运营人员、数据分析师、python工程师、数据挖掘工程师。
那么学习Python数据分析，你不得不学习的一个模块，就是Pandas！
Pandas是啥？
Pandas是一个强大的分析结构化数据的工具集；它的使用基础是Numpy（提供高性能的矩阵运算）；用于数据挖掘和数据分析，同时也提供数据清洗功能。
为啥学Pandas？
- 数据展示极简
Pandas 提供了极其简化的数据表示形式。这有助于更好地分析和理解数据。更简单的数据表示有助于数据科学项目获得更好的结果。
- 书写逻辑清晰，功能强大
这是 Pandas 的最大优势之一。在没有任何支持库的情况下，在 Python 中需要多行代码，但使用 Pandas 只需 1-2 行代码即可实现。因此，使用 Pandas 有助于缩短处理数据的过程。节省了时间，我们可以更多地关注数据分析算法。
总结：
Here are just a few of the things that pandas does well:
- Easy handling of missing data (represented as NaN, NA, or NaT) in floating point as well as non-floating point data
- Size mutability: columns can be inserted and deleted from DataFrame and higher dimensional objects
- Automatic and explicit data alignment: objects can be explicitly aligned to a set of labels, or the user can simply ignore the labels and let Series, DataFrame, etc. automatically align the data for you in computations
- Powerful, flexible group by functionality to perform split-apply-combine operations on data sets, for both aggregating and transforming data
- Make it easy to convert ragged, differently-indexed data in other Python and NumPy data structures into DataFrame objects
- Intelligent label-based slicing, fancy indexing, and subsetting of large data sets
- Intuitive merging and joining data sets
- Flexible reshaping and pivoting of data sets
- Hierarchical labeling of axes (possible to have multiple labels per tick)
- Robust IO tools for loading data from flat files (CSV and delimited), Excel files, databases, and saving/loading data from the ultrafast HDF5 format
- Time series-specific functionality: date range generation and frequency conversion, moving window statistics, date shifting and lagging
Pandas相关资料
1. 官网
https://pandas.pydata.org/
2. Cheatsheet
https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
3. 快速入门
https://www.pypandas.cn/docs/getting_started/
这篇文章在讲啥？
那么这一篇文章是在讲啥呢？
不好意思，这不适合一个完全没用过Pandas甚至对Python语法一点了解没有的数据小白。
本篇文章适合：
1. 有一定Python语言基础的同学
2. 使用过EXCEL的同学
3. 了解数据分析基础的同学
那么，这里主要是为大家梳理在日常使用Pandas处理数据的时候，可能需要注意的一些关键点！
同时也是面试（主要是take home project）、甚至以后工作会遇到的！
---
正式内容
ok，那我们就正式开始了！
import pandas as pd
import numpy as np
1. DAY1（读取文件）
1.1 读取csv
读取csv是我们最常见的情况，大家好好看看！
详见：http://xhslink.com/bYaeVh
1.2 读取json
Use read_json() to read directly from a URL into a DataFrame
Json是一个应用及其广泛的用来传输和交换数据的格式，它被应用在数据库中，也被用于API请求结果数据集中。虽然它应用广泛，机器很容易阅读且节省空间，但是却不利于人来阅读和进一步做数据分析，因此通常情况下需要在获取json数据后，将其转化为表格格式的数据，以方便人来阅读和理解。常见的Json数据格式有2种，均以键值对的形式存储数据，只是包装数据的方法有所差异。
建议阅读：
1. https://www.kaggle.com/code/tboyle10/working-with-json-files/notebook
2. https://juejin.cn/post/6994208008167227406
# load pandas and json modules                                                                                               
import pandas as pd
import json

# json string                                                                                                                
s = '{"col1":{"row1":1,"row2":2,"row3":3},
      "col2":{"row1":"x","row2":"y","row3":"z"}}'

# read json to data frame                                                                                                    
df = pd.read_json(s)
print(df)
1.3 读取html
获取网页上的表格数据
详见：https://www.cnblogs.com/litufu/articles/8721207.html
[图片]
url = "https://fund.eastmoney.com/fund.html#os_0;isall_0;ft_;pt_1"
table = pd.read_html(url, attrs = {'id': 'oTable'}, header=0)
type(table) 
# list
len(table)
# 1
table[0]
1.4 DAY1习题
1. 读取数据（链接），并且随机跳过（skip）90%的行（seed=1）
期待输出：
[图片]
2. 读取数据（链接），并随机从0-100行中挑选10行去掉（seed=2022）
期待输出：
[图片]
3. 读取数据（链接），爬取前5页数据
[图片]
---
2. DAY2（系列Series）
pd.Series( data, index, dtype, name, copy)
Series 与 NumPy 数组非常相似（实际上构建在 NumPy 数组对象之上）。
NumPy 数组与 Series 的区别在于，Series 可以具有轴标签，这意味着它可以由标签索引，而不仅仅是数字位置。
它也不需要保存数字数据，它可以保存任意 Python 对象。
2.1 创建Series
'''
将list、numpy array 或 dict 转换为 Series
'''

labels = ['a','b','c']        
my_list = [10,20,30]         # 列表
arr = np.array([10,20,30])   # 数组
d = {'a':10,'b':20,'c':30}   # 字典
pd.Series(data=my_list,index=labels)   
# 不设置index默认自然数id递增
pd.Series(arr,labels)
pd.Series([sum,print,len])  # 里面甚至可以用函数

# 自己试试吧
s = pd.Series([1,2,3],index = ['A','B','C'],name = 'First_Series')
dates = pd.date_range("20220101", periods=6)
2.2 Series的性质
s = pd.Series([1,2,3],index = ['A','B','C'],name = 'First_Series')

# 如何查看对象的性质呢？
# 1. 查看系列的值
s.values
# array([1, 2, 3], dtype=int64)
# 返回数组

# 2. 查看系列的名称
s.name
# 'First_Series'

# 3. 查看系列的索引
s.index
# Index(['A', 'B', 'C'], dtype='object')
[图片]
2.3 系列的计算
ser1 = pd.Series([1,2,3,4],index = ['北京', '上海','深圳', '广东'])  
ser2 = pd.Series([1,2,5,4],index = ['北京', '西藏','深圳', '新疆'])  

# 他们加起来会发生什么呢？
ser1 + ser2
[图片]
2.4 DAY2习题
- 从列表、字典、数组创建Series
l = [0, 1, 2, 3, 4]
d = {'a':1,'b':2,'c':3,'d':4,'e':5}
arr = np.array([1,2,3,4,5])
- 创建一个series（叫做s即可），值为小于100的所有偶数，index是20220101开始的日期，后续为每个月初（如下面结果所示），序列名字设置为whale
# 你代码的输出结果前五行应该跟下面的一模一样！

2022-01-01     0
2022-02-01     2
2022-03-01     4
2022-04-01     6
2022-05-01     8
- 写出一个函数series_info
  - 入参：series
  - 出参：打印系列的名称、长度、数据类型、索引、值
  - 测试用例：上一题的series
def series_info(series):
    s = series.copy()
    print(f'系列的名称：{你的代码}')
    print(f'系列的长度：{你的代码}')
    print(f'系列的数据类型：{你的代码}')
    print(f'系列的索引：{你的代码}')
    print(f'系列的值：{你的代码}')
    return 
[图片]
---
3. DAY3（数据框DataFrame基础）
pandas.DataFrame( data, index, columns, dtype, copy)
DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
就当做excel的sheet或者sql的table来看就好，有行有列，也是最常见的数据结构！
[图片]
[图片]
3.1 创建数据框（df）
很多同学对于表格很熟悉，但是让你用代码敲出一个df，有的同学就懵逼了，别看这个基础就忽略它的重要性！
这一part学习两大块内容
- 如何创建df
- df基操
---
1. 列表创建（Create dataframe from list of lists）
data = [['A',10],['B',12],['C',13]]
# 这里的['A',10]代表每一行的内容，共有三行
df = pd.DataFrame(data,
                  columns=['第一列','第二列'], 
                  index = ['第一行','第二行','第三行'])
# columns设置列名
# index设置行名（索引）
df
[图片]
2. 数组创建（ndarray）
data = np.array([[11,22,33],[44,55,66]])
df = pd.DataFrame(data)
df
# 如果不自定义columns和index，会用递增自然数默认填充！
[图片]
3. 字典创建（Create dataframe from dictionary of lists）
data = {'名字':['小张','小王','小李','小赵'],
        '年龄':[23,22,21,24]}
df=pd.DataFrame(data)
df
# 字典的方法，我觉得是最好的，希望大家掌握！
[图片]
4. 组合创建
df = pd.DataFrame({  'A' : 1.,    
                     # 使用1.代表1.0，用一个浮点数填充整个列
                     'B' : pd.Timestamp('20130102'),
                     # 用时间戳填充整个列
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     # 用系列
                     'D' : np.array([3]*4,dtype='int32'),
                     # 用数组
                     'E' : pd.Categorical(["test","train","test","train"]),
                     # 转换为类别变量
                     'F' : 'foo' })
                     # 字符填充
df
# 一定要保证每个列的长度是一样的！
[图片]
3.2 数据框操作基础
1. 查看前n行
df.head(n)
2. 查看后n行
df.tail(n)
3. 查看df的index
df.index
4. 查看df的columns
df.columns
5. 查看shape（几行几列）
df.shape
6. 查看size（有多少数据）
df.size
7. 查看维数
df.ndim
8. 查看数据类型
df.dtypes
9. 查看df的值（转换为数组）
df.values
df.to_numpy()
10. 对df快速统计汇总（连续型变量）
df.describe()
11. 转置（行转列，列转行）
df.T
12. 对轴排序
df.sort_index(axis=0或1, ascending=False)
13. 对值排序
df.sort_values(by='列名')
那么，我们来举个例子，大家就明白了！
首先创建一个随机数组，然后转换为df！
df = pd.DataFrame(np.random.randn(100,4), columns=list('ABCD'))
df
[图片]
3.3 DAY3习题
1. 创建一个10行4列的随机整数数矩阵（叫df）（-10到10）（np.random.randint），列名为a、B、c、D（使用split），行索引为20220101开始的每五天（每隔四天）的timestamp。(seed=2022)
期待输出：
[图片]
2. 对上面的df做以下操作（不得改变原数据，不得make copy）：
  1. 关于列名从大到小排序（忽略大小写）
  2. 再接着，对c这一列的值的平方进行排倒序，如果平局对D列继续排序，然后是a，最后是B
  3. 最后忽略掉行index的变化
期待输出：
[图片]
3. 写一个函数view_df
  1. 输入为df和sort_by
    1. df
    2. sort_by指的是根据某个列进行排序（正序）
  2. 输出为此df的前4行和后4行合在一起（pd.concat）
[图片]
  带入第一题的df，如果对view_df(df）的结果关于不同的列的值进行排序（正序），请统计一下排序后第一行数据对应的时间是在上半个月（该日期在15号之前）的case的个数【答案是3】（是的，故意在难为大家，自己梳理梳理分解一下）
---
4. DAY4（数据框DataFrame列操作）
这一part继续上回书说到，我们介绍df的高阶操作！
（内容较多，耐心看！）
切片&索引
- 取一列或者多列（基础）
- 构造新列
- 插入新列
- 移除列
df = pd.DataFrame(np.random.randn(10,4), 
                  columns=list('ABCD'),
                  index = pd.date_range('20130101', periods=10))
df
[图片]
4.1 取一列或者多列
[图片]
1. df.列名
2. df['列名']
3. df[['列名1','列名2']]
4. df.loc[:,'列名']   
5. df.iloc[:,0]
# 取A列
df.A 
df['A']

# loc和iloc，重要！
df.loc[:,'A']
df.iloc[:,0]
# 冒号代表取所有行
# 逗号后面代表我们要取的列
    # loc 输入的是列名
    # iloc 输入的是第几列（数字）

# 取A和B这两列
df[['A','B']]
df.loc[:,['A','B']]
df.iloc[:,[0,1]]
4.2 构造一个新列
1. df['新列名'] = df['列1']+df['列2']
2. df['新列名'] = 固定值
3. df.reindex
4. pd.concat
5. df.loc[:,'新列名']
4.3 插入新列（assign vs. insert）
- Assigin
  - 构造多个新列，并且新列之间的计算逻辑相关联
  - 返回一个new object，不做原地操作
  - 不能插入指定位置
- Insert
  - 原地操作
  - 可以插入指定位置
  - 如果要插入已存在的列名，需要设置allow_duplicates = True（不建议）
# 养成copy的好习惯，不然很可能把原来的数据覆盖！
df1 = df.copy()
# assign
df1.assign(new1 = [1,2,3,4,5,6,7,8,9,10])
# assign默认把新列加到最后一列
# 注意
# 其实上面的代码是对df1的copy进行了操作
# 而没有赋值于df1本身
[图片]
[图片]
# 养成copy的好习惯，不然很可能把原来的数据覆盖！
df1 = df.copy()
# insert
df1.insert(1, "new2", [1,2,3,4,5,6,7,8,9,10])
# 表示把new2这列插入到第一列的后面
df1
# 注意
# df1被改变了！
[图片]
4.4 移除列
# 去除E列
df.drop('E',axis=1)
df
# 发现df没有被改变
df.drop('E',axis=1, inplace=True)
# 发现df改变了，inplace=True表示对df原地操作
[图片]
4.5 DAY4习题
1. 创建一个10行1000列的随机数矩阵（np.random.randn），列名为1开始的整数（1到1000），行索引为20220101-20220110。（seed=1）
2. 取出上面df的偶数列（第n列，n是偶数），命名为df1
3. 把df1按照列名从大到小排序（不用sort_index）
4. 在df1的列名中，按照从小到大排序，第19个被7整除的值，在那一列的后面插入一个新列叫new，值为df的偶数列的均值（axis=1）减去df奇数列的均值。
---
5. DAY5（数据框DataFrame行操作）
上一节我们说到列的操作，那么我们一样可以对行进行操作！
（内容较多，耐心看！）
- 取一行或者多行
- 添加新行
- 插入新行
- 移除行
5.1 取一行或者多行
np.random.seed(1)
df = pd.DataFrame(np.random.randn(10,4), 
                  columns=list('ABCD'),
                  index = pd.date_range('20130101', periods=10))
df
[图片]
### iloc or loc ###
df.loc['2013-01-03']  # 筛选索引名称是'2013-01-03'的行
df.iloc[1]            # 筛选第二行

df.loc['2013-01-03':'2013-01-06']  
# 筛选索引名称是'2013-01-03'至'2013-01-06'的行
df.iloc[3:6]
# 筛选第4行到第6行（左闭右开）
5.2 添加行
# 1. loc
df1 = pd.DataFrame(np.random.randint(0,10,(5,4)),
                   columns = 'A B C D'.split())
df1.loc[len(df1.index)] = 4
# 后续介绍其他方法
[图片]
5.3 插入新行
新增一行貌似还不难，但是在指定位置加一行，其实是没有直接可以调用的方法帮助我们实现的，这里大家需要好好听讲！
# 1. loc
df1 = pd.DataFrame(np.random.randint(0,10,(5,4)),
                   columns = 'A B C D'.split())
                   
# 在index为1和2之间加入一行，但是只能插入在最下面（下面左图）
df1.loc[1.5] = [1,2,3,4]   
# 根据index进行排序，那么1.5肯定排在1和2之间
# 然后再重新设置index，并把原来的index给移除（drop）         
df1.sort_index().reset_index(drop=True,inplace=True)
[图片]
[图片]
5.4 移除行
这里可以用drop函数，跟移除列类似
DataFrame.drop(labels=None, 
               axis=0, 
               index=None, 
               columns=None, 
               level=None, 
               inplace=False, 
               errors='raise')
这里默认axis=0就是移除行的意思，labels则输入行名即可。
# 1这里指的是labels=1，也就是移除行名为1的行
df1.drop(1)
5.5 DAY5习题
np.random.seed(1)
df = pd.DataFrame(np.random.randn(1000,4), 
                  columns=list('ABCD'),
                  index = pd.date_range('20200101', periods=1000))
针对上面的df，请完成：
1. 取所有日期为偶数的行
[图片]
2. 对df后面插入5行数据，index要连续，具体插入的数据是1,2,3,4,5，如题所示
[图片]
3. 如果是插入n（比如100）行呢？
[图片]

---
6. DAY6（行列组合操作）
那么同时对行列进行操作呢？这里需要大家重点掌握！
np.random.seed(1)
df = pd.DataFrame(np.random.randn(10,4), 
                  columns=list('ABCD'),
                  index = pd.date_range('20220101', periods=10))
6.1 行列切片
# loc

# 行名为2022-01-03，列名为A的值是多少
df.loc['2022-01-03','A']
# 行名为2022-01-03，列名为A和C的值是多少
df.loc['2022-01-03',['A','C']]
# 行名为2022-01-03及之后所有的行，列名为A和C的值是多少
df.loc['2022-01-03':,['A','C']]

# iloc

# 第3行，第3列
df.iloc[2,2]
# 第3行及之后的行，第2行以及之前的行（左闭右开）
df.iloc[2:,:2]
6.2 行列筛选（进阶）
如果我们需要根据一些自定义的逻辑去对df进行筛选呢？
单条件
比如：筛选所有A列大于0的行
[图片]
[图片]
多条件
比如：
- 筛选所有A列大于0并且C列小于0的行
- 筛选所有A列大于0.5或者D列小于-1的行
[图片]
[图片]
query
pandas中的query方法能更简洁的帮助我们对表格查询！看看它的语法吧，很像sql！
比如，如何实现上述的那些需求？我们用query试试！
query的引号里面的语法类似于sql的查询，多条件筛选时可读性强，并且简洁。
[图片]
[图片]
[图片]
query也支持使用变量
[图片]
isin
isin方法传入列表，相当于对isin前面的内容和后面的匹配，成功则返回bool值。
[图片]
[图片]
其中，~的意思是not，也就是筛选过程中的不选的意思。
nlargest or nsmallest
根据某列取其前n个最大值（最小值）
[图片]
[图片]
6.3 DAY6习题
np.random.seed(1)
df = pd.DataFrame(np.random.randn(1000,4), 
                  columns=list('ABCD'),
                  index = pd.date_range('20220101', freq = 'MS',periods=1000))
df
[图片]
1. 找出同时满足下列条件的数据
  1. 月份的英文字母第一个是M的所有行
  2. A列大于0
  3. C列小于0
  4. 按照D列从大到小排序，取前10行
[图片]
2. 找出满足A+B大于n并且C-D大于n的行数
其中n为-1到1的等距数组（10）
[图片]

---
7. DAY7（数据抽样）
[图片]
这里使用Iris数据集，为大家介绍如何进行数据的抽样，先简单介绍一下这个数据集。
简单说，该数据集由3种不同类型的鸢尾花的各50个样本数据构成。其中的一个种类与另外两个种类是线性可分离的，后两个种类是非线性可分离的。
如何引入数据？
直接copy下面代码即可！
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
7.1 随机采样
给定一个df为 N 行，那么我们如果想随机抽 X 行，应该怎么做？
其中$$X \leq N
$$。
df.sample(5)
# 当你每次run这行代码的时候，你都会得到不一样的值
# 因为你没有设置种子（random_seed）
[图片]
# 每一次run这一chunk，都不会变
df.sample(n=5,random_state = 1)
[图片]
df.sample(n=100,
          random_state = 1,
          replace=True)   # replace = True 代表可以重复抽样
          
# 检验
# df.sample(n=100,
            random_state = 1,
            replace=True).duplicated().sum()
# 可以看出27%的行数被重复筛选了！
df.sample(frac=0.5,        # frac 代表抽样的比例
          random_state = 1,
          replace=True)   
7.2 加权抽样
要知道这里面还有一个target没有引入，其中
data.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
data.target
# 0 ==> setosa
# 1 ==> versicolor
# 2 ==> virginica
那么，我们想对不同的花（species不同）进行加权抽样，比如：
- 抽到setosa（target=0）的概率为20%
- 抽到versicolor（target=1）的概率为50%
- 抽到virginica（target=2）的概率为30%
# 在df中新建一列叫weights，其值为target所对应的占比
# map的作用是把值进行映射（不会不要紧，先了解用法）
df['weights'] = pd.Series(data.target).map(
    {
        0:20,   # 0变成20
        1:50,   # 1变成50
        2:30    # 2变成30
        # 加起来为100，也就是各个权重为20%、50%、30%
    }
)
df.sample(n=100,random_state = 1,weights = 'weights')
[图片]
那么如何证明我们的抽样是按照权重抽的呢？
df.sample(n=100,
          random_state = 1,
          weights = 'weights')['weights'].value_counts()/100
          
# 50    0.38   50%
# 30    0.33   30%
# 20    0.29   20%
# Name: weights, dtype: int64
可以看出，貌似没达到我们预期的抽样权重啊，这是为什么呢？（见作业第三题）
那么如果我们不想加一列再加权抽样，直接抽不行么？
当然可以！
df.sample(n = 100,
          weights = [20]*50+[50]*50+[30]*50,
          random_state=1)
          
# 你可以把结果和上面的比对一下，完全一样哦！
7.3 DAY7习题
1. 读取数据，链接在这里。
2. 对该数据进行加权抽样（n = 100，seed=1）：
  1. Iris-versicolor权重为70%
  2. Iris-virginica权重为20%
  3. Iris-setosa权重为10%
输出：
[图片]
3. 解释上面内容中为什么没达到我们预期的抽样权重，如何才能证明我们的代码是合理的？
提示：
[图片]
4. 从三个类别中分别抽取5条数据，也就是150条抽15条出来（不用group by）（seed=1）
输出：
[图片]
5. 随机抽取80%的数据作为训练集，命名为train，剩下的20%作为测试集。（seed=12345）
如果你发现test在notebook中全部行都展示出来了，想办法变为下面右图的样子！
下图左面是train，右面是test。

[图片]
[图片]
---
ok，前7天的课程内容就到这里啦~
希望大家多多复盘和总结，有问题记得跟我约时间oneone！
---
补充内容：常用option设置
官方网站链接
暂时无法在飞书文档外展示此内容
