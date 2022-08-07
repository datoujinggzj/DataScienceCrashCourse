## 前言

哈喽大家好，我是不卖焦虑，只聊干货的鲸鲸！

很多小伙伴问我，是不是会sql、excel还有可视化工具（Tableau、powerBI）就可以成为一个数据分析师了，我觉得其实可以，但是得看你想成为一个什么样的数据分析师。
想找个工作不是很难，想找个你满意的数据分析工作还是挺难的。


<div align=center>
<img src=../img/image.png width='200' />
</div>

目前Python数据分析适用于众多行业，包括并不局限于网站运营、销售竞争、新媒体传播、互联网公司对数据的分析等。Python数据分析人员主要担任的岗位：企业运营人员、数据分析师、python工程师、数据挖掘工程师。
那么学习Python数据分析，你不得不学习的一个模块，就是Pandas！

### Pandas是啥？
Pandas是一个强大的分析结构化数据的工具集；它的使用基础是Numpy（提供高性能的矩阵运算）；用于数据挖掘和数据分析，同时也提供数据清洗功能。

### 为啥学Pandas？
- 数据展示极简
Pandas 提供了极其简化的数据表示形式。这有助于更好地分析和理解数据。更简单的数据表示有助于数据科学项目获得更好的结果。
- 书写逻辑清晰，功能强大
这是 Pandas 的最大优势之一。在没有任何支持库的情况下，在 Python 中需要多行代码，但使用 Pandas 只需 1-2 行代码即可实现。因此，使用 Pandas 有助于缩短处理数据的过程。节省了时间，我们可以更多地关注数据分析算法。


### 小结

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

### Pandas相关资料

1. 官网
https://pandas.pydata.org/
2. Cheatsheet
https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
3. 快速入门
https://www.pypandas.cn/docs/getting_started/


### 这篇文章在讲啥？

那么这一篇文章是在讲啥呢？

不好意思，这不适合一个完全没用过Pandas甚至对Python语法一点了解没有的数据小白。

本篇文章适合：
1. 有一定Python语言基础的同学
2. 使用过EXCEL的同学
3. 了解数据分析基础的同学

那么，这里主要是为大家梳理在日常使用Pandas处理数据的时候，可能需要注意的一些关键点！
同时也是面试（主要是take home project）、甚至以后工作会遇到的！

---

### 干货在此！！！

- [第1周文档](https://ex661wn4s4.feishu.cn/docx/doxcnYhnPWtZBw9ceJZGG0wsQTf)
- [第2周文档](https://ex661wn4s4.feishu.cn/docx/doxcnuqfIOQwgKayh8O4XwckjAC)

|  日期   |   主要内容  |  链接   |
| :---: | :--------: | :---: | 
|  DAY1  |   读取文件  |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY1/Pandas_DAY1.ipynb)   |     |
|   DAY2  |  系列Series   |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY2/Pandas_DAY2.ipynb)   |     
|   DAY3  |  数据框DataFrame基础   |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY3/Pandas_DAY3.ipynb)    |     
|   DAY4  |  数据框DataFrame列操作   |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY4/Pandas_DAY4.ipynb)    |     
|  DAY5  |  数据框DataFrame行操作  | [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY5/Pandas_DAY5.ipynb)    |     
|  DAY6   |  行列组合操作   | [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY6/Pandas_DAY6.ipynb)     |     
| DAY7   |  数据抽样（补充：常见`option`设置）   |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY7/Pandas_DAY7.ipynb)    |     
|  DAY8   |  `value_counts`全解（补充：`style.format` 使用）   |   [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY8/Pandas_DAY8.ipynb)  |     
| DAY9   |  合并&拼接   |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY9/Pandas_DAY9.ipynb)   |     
|  DAY10  |  group by   | [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY10/Pandas_DAY10.ipynb)      |     
|  DAY11  |  五大函数   |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY11/Pandas_DAY11.ipynb)    |     
| DAY12 |   窗口函数  |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY12/Pandas_DAY12.ipynb)   |    
|  DAY13   |   时间处理  |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY13/Pandas_DAY13.ipynb)     |    
| DAY14 |   缺失值处理  |  [代码](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DAY14/Pandas_DAY14.ipynb)   |     
