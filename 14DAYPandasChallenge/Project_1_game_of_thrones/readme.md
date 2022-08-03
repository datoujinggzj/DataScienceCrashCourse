# 项目练习（1）

### 一、简介

《权力的游戏》大致是根据小说《冰与火之歌》的故事线情节，故事背景设置于虚构的维斯特洛七王国及厄斯索斯大陆。 该系列记录了该领域贵族争夺铁王座的激烈王朝斗争，而其他家庭则为争取独立而斗争。

### 二、资源获取

#### 数据

- [battles](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DATA/battles.csv)
- [character-deaths](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DATA/character-deaths.csv)
- [character-predictions](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/DATA/character-predictions.csv)

#### 作业

- [作业链接](https://gitee.com/gzjzg/data-preparation-crash-course/blob/master/14DAYPandasChallenge/Project_1_game_of_thrones/Project_1_game_of_thrones.ipynb)

### 三、数据集介绍
该数据集包含了三个CSV文件，分别是：battles.csv，character-deaths.csv和character-predictions.csv。现分别介绍如下：

battles.csv：《权利的游戏》中所有的战争。
character-deaths.csv：人物死亡数据集。
character-predictions.csv：人物命运预测数据集。
在每一个CSV数据里面都包含了大量的变量，读懂这些变量名所代表的含义非常有必要。例如：battles.csv中的year变量代表战争发生的时间，battle_type代表战役类型，有伏击战，围攻战，突袭战等。

### 四、英文介绍

#### Context

Game of Thrones is a hit fantasy tv show based on the equally famous book series "A Song of Fire and Ice" by George RR Martin. The show is well known for its vastly complicated political landscape, large number of characters, and its frequent character deaths.

#### Content

Of course, it goes without saying that this dataset contains spoilers ;)

This dataset combines three sources of data, all of which are based on information from the book series.

- Firstly, there is battles.csv which contains Chris Albon's "The
War of the Five Kings" Dataset. Its a
great collection of all of the battles in the series.

- Secondly we have character-deaths.csv from Erin Pierce and Ben
Kahle. This dataset was created as a part of their Bayesian Survival
Analysis.

- Finally we have a more comprehensive character dataset with
character-predictions.csv. It
includes their predictions on which character will die.

#### Acknowledgements

- Firstly, there is battles.csv which contains Chris Albon's "The war of the Five Kings" Dataset, which can be found here:

https://github.com/chrisalbon/war_of_the_five_kings_dataset . 

It's a great collection of all of the battles in the series.

- Secondly we have character-deaths.csv from Erin Pierce and BenKahle. This dataset was created as a part of their Bayesian Survival
Analysis which can be found here: http://allendowney.blogspot.com/2015/03/bayesian-survival-analysis-for-game-of.html

- Finally we have a more comprehensive character dataset with character-predictions.csv. This comes from the team at A Song of Ice and Data who scraped it from http://awoiaf.westeros.org/ . It also includes their predictions on which character will die, the methodology of which can be found here: https://got.show/machine-learning-algorithm-predicts-death-game-of-thrones

- Inspiration

What insights about the complicated political landscape of this fantasy world can you find in this data?
  
  
- https://asoiaf.fandom.com/zh/wiki/%E6%97%8F%E8%AF%AD%E5%88%97%E8%A1%A8?variant=zh-hk