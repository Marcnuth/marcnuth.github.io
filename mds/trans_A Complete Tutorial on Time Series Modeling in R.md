#基于R语言的时间序列模型完全指南
A Complete Tutorial on Time Series Modeling in R
作者: TAVISH SRIVASTAVA
译者: Marcnuth


##简介

时间是确保商业成功的重要因素，但我们却很难跟上时间的脚步。但是，技术的
发展让我们能一窥时间内部的细节，并提前预知某些事情的发生。别担心，我不
是要谈论跨时空的时间机器，而是谈论一些能预测未来的技术方法。而这其中的
一种办法，就是利用基于时间的时间序列模型。

正如这个名字本身暗含的意思一样，它所囊括的数据时基于时间的，这个时间可
以是年月日，也可是时分秒。我们也是利用这些基于时间的数据，来进行针对未
来的预测，并帮助我们作出决定。

当你拥有一系列相关联的时间序列数据时，你会发现时间序列模型的妙处。现实
中，我们也有很多时间序列模型的应用例子，比如，预测来年的销售额，预测网
页未来的浏览量，预测比赛的排名等等。然而，这些领域，也是目前很多分析师
不能理解和处理的。

因此，如果你对时间序列模型的处理过程不甚了解，那么这个教程就是为你而写
的，在这个教程里，我们会介绍各种各样的时间序列模型，并介绍相关的技术。
下面是内容列表：

- 基础： 时间序列模型
- 用R处理时间序列模型
- ARMA时间序列模型介绍
- ARIMA模型的框架和应用

让我们开始吧!

## 基础：时间序列模型

让我们先介绍一些基础知识。本节会包括序列平稳性、随机游走、斯皮尔曼相关
系数（Rho Coefficient）、迪基-福勒检验（Dickey Fuller Test of
Stationarity）。看到这些名字不要害怕，我保证你一步步看下去能逐渐明晰这
些概念，并享受这个过程。

### 序列的平稳性

一般而言，有三个标准来判断一个时间序列是不是平稳的:
- 1. 时间序列的均值应该是一个常量，即不随时间的改变而改变。下图中，左
图满足这个条件:
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Mean_nonstationary.png)
- 2. 时间序列的方差应该是一个常量，也不随时间的改变而改变。这就是常说
的同方差性(homoscedasticity)。下图中左图满足这个条件:
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Var_nonstationary.png)
- 3. 序列中，第i项和第i+m项的相关系数不随时间的改变而改变。下图中左图满
足这个条件::
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Cov_nonstationary.png)

### 我们为什么关注时间序列的平稳性?

我们关注时间序列的平稳性的原因在于，除非你的时间序列是平稳的，不然你不
能用这些时间序列数据去建立相关的模型。尽管有时我们也会违背这条准则，但
是那是因为在一开始的时候，我们就采用了某些办法去使得时间序列变得平稳。
通常情况下，可以采用去除趋势或者求差分等方法来使得时间序列变得平稳。

### 随机游走

这也是时间序列中一个非常基础的概念，你可能已经了解了。但是，我发现工业
界的很多人会将这个概念和随机过程混淆。本节中，我将借助一些数学方法，来
使得这个概念变得清晰起来。首先来看个例子吧：

**例子**：想象一个女孩在一个超级大的棋盘上随机的移动。在这种情况下，女
孩下一步走到哪里仅仅与她上一步走到哪里有关系。
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/RandomWalk.gif)

试想，你现在坐在另外一个房间并且看不到这个女孩。你想要预测这个女孩在不
同时间的位置。你能预测到什么样的精度呢？事实上，随着时间推移，你的预测
结果将会越来越不精确。在t=0的时候，你完全清楚她在哪里，但是接下来，她
可以向相邻的8个方块移动，因此你的预测准确率只有1/8。现在，让我们试着写
下这个时间序列:

```
X(t) = X(t-1) + Er(t)
```

其中，Er(t)是在t时刻的误差，这是女孩每次移动带来的随机噪声。
现在，如果我们递归的替换X，我们最终会得到这样一个等式：

```
X(t) = X(0) + Sum(Er(1),Er(2),Er(3).....Er(t))
```

现在，让我们试着去检验一样这是不是一个平稳的时间序列。

#### 1. 均值是常量吗？


```
E[X(t)] = E[X(0)] + Sum(E[Er(1)],E[Er(2)],E[Er(3)].....E[Er(t)])
```

我们知道，随机噪声的均值是0，因此:
E[X(t)] = E[X(0)] = 常量

#### 2. 方差是常量吗？

```
Var[X(t)] = Var[X(0)] +
Sum(Var[Er(1)],Var[Er(2)],Var[Er(3)].....Var[Er(t)])

Var[X(t)] = t * Var(Error) = Time dependent.
```

因此，我们可以推断，随机游走并不是一个随机过程，因为它的方差并不是常量。
你也可以试试计算它的相关系数，你会发现，它的协方差也是随着时间变化的。


### 进一步探索: 让我们增加点趣味吧

截止目前，我们已经知道，随机游走并不是平稳的时间序列。那么让我们引入一
个新的相关系数到这个方程中，试试看能不能让它变成平稳的。

#### 引入相关系数: 斯皮尔曼相关系数(Rho)

```
X(t) = Rho * X(t-1) + Er(t)
```

现在，让我们通过改变斯皮尔曼系数来使得序列平稳。本节中，我将用图形来解
释相关内容，不做平稳性的校验。

首先，如果我们使得斯皮尔曼系数为0，将得到一个随机噪声，它是一个完美的
平稳的时间序列。

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho0.png)


接下来，让我们把斯皮尔曼系数改到0.5试试看:

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho5.png)

你可以已经注意到，这个序列的某些“循环”变得更宽。但是和平稳的序列相比，
还是看不出明显的差别。让我们试试把系数改到0.9呢:

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho9.png)

我们可以看到，X(t)的值变得没那么极端的，但是和平稳序列比起来，似乎差别
还是不是很大。现在，让我们把系数设置为1：

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho1.png)

非常明显的，这个和平稳的时间序列完全不一样。那么是什么导致时间序列的图
发生了如此明显的变化?让我们回头看看之前的数学公式:

```
X(t) = Rho * X(t-1) + Er(t)
```

取均值后:

```
E[X(t)] = Rho *E[ X(t-1)]
```

从这个方程，我们就可以看到，对于t时刻的值X(t), 会被“拉”到和上一时刻的
值。

举个例子:

如果X(t-1)=1， E(X(t)) = 0.5。现在，如果X移到了远离0的地方，那么下一步，
它将会被往0的方向“拉”。 唯一能让它继续往离0更远的方向偏移的因素就只有
误差项了，但是误差项往任何方向偏离的概率都是一样的。所以，斯皮尔曼系数
为1的时候意味着: No force can pull the X down in the next step.


#### 迪基-福勒平稳性检验

事实上，上一节我们讲的其实已经就是迪基福勒检验了。在上一节的公式上做一
点小变换，我们就能得到迪基福勒检验的数学等式了:

```
X(t) = Rho * X(t-1) + Er(t)
```

```
=>  X(t) - X(t-1) = (Rho - 1) X(t - 1) + Er(t)
```

通过检验 Rho - 1 是否显著，如果原假设被拒绝，那么这个时间序列就是一个
平稳的时间序列。

平稳性检验和将一个序列转化为平稳的时间序列，在时间序列建模的过程中是及
其重要的。在学习下一章之前，你需要记清楚这些概念。

## 用R处理时间序列数据

本章的目的是学习用R去处理时间序列数据，需要注意的是，我们在本章只谈如
果处理数据，不谈如何进行建立数据的模型。

R中有一个内置的数据集：AirPassengers。这个数据集包括了从1949-1960年的
乘客数量月度数据，将会被我们使用。

### 加载数据集

参见下面的代码:

```
> data(AirPassengers)
> class(AirPassengers)
	[1] "ts"
#This tells you that the data series is in a time series format
> start(AirPassengers)
	[1] 1949 1
#This is the start of the time series
> end(AirPassengers)
	[1] 1960 12
#This is the end of the time series
> frequency(AirPassengers)
	[1] 12
#The cycle of this time series is 12months in a year
> summary(AirPassengers)
	Min. 1st Qu. Median Mean 3rd Qu. Max.
	104.0 180.0 265.5 280.3 360.5 622.0
```


### 常用命令

```
#The number of passengers are distributed across the spectrum
> plot(AirPassengers)
#This will plot the time series
> abline(reg=lm(AirPassengers~time(AirPassengers)))
# This will fit in a line
```

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/plot_AP1.png)

还有一些能让你更了解数据细节的命令:

```
> cycle(AirPassengers)
#This will print the cycle across years.
>plot(aggregate(AirPassengers,FUN=mean))
#This will aggregate the cycles and display a year on year trend
> boxplot(AirPassengers~cycle(AirPassengers))
#Box plot across months will give us a sense on seasonal effect
```

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/plot_aggregate.png)
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/plot_month_wise.png)


### 结论

从上面我们可以看到:
- 乘客数量逐年递增
- 7、8月的方差均值比其他月份更高
- 尽管每个月的均值都不一样，但是它们的方差变化却很小。因此，这个数据有
着强烈地季节效应，周期为12或者更少。


在建模之前，深入探索一下数据是很重要的，不然，你无法知道这个数据是否是
平稳的。在这个例子中，我们已经知道了很多数据的细节。


## 用ARMA对时间序列数据建模

ARMA模型在时间序列处理中用的十分广泛。其中，AR代表自回归，而MA代表移动
平均。我知道这些名字听起来让人难以理解，别担心，我会在下面用比较简单地
方式来介绍它们。

现在，我们需要掌握这些名词关联的技能，并且理解不同模型的特性。但是在此
之前，你应当牢记，AR和MA模型不能应用于非平稳的时间序列。

那么，接下来我将先分别介绍AR和MA模型，然后在一同探讨一下这些模型的特性。

### AR自回归模型

举个例子:

用X(t)代表一个国家当前的GDP，且依赖于上一年的GDP X(t-1)。

因此，我们可以得到:

```
x(t) = alpha *  x(t – 1) + error (t)
```

这个方程其实就是AR(1)模型。数字1代表了当前值仅依赖于上一个值。alpha是
我们需要求解的参数，以最小化模型和数据的误差。值得注意的，根据上面等式，
X(t-1)也依赖于X(t-2)。因此，X(t)的值对于未来的值的影响将越来越小。

举个例子：假设X(t)是某个城市某天卖出的果汁数量。在冬天，购买果汁的顾客
少了。但是突然某一天，气温上升，果汁的销售量又飙升到1000，然后之后的几
天内，温度又变得冷起来。即便是这样，偶然飙升的销售量并不会对模型有什么
影响，我们也知道，人们都喜欢在热的天气下买果汁喝，只有50%的人在寒冷的
情况下依旧会买，之后，随着温度降低，这部门人越来越少，可能只有25%（50%
的50%）的人会继续购买。下图展示了满足AR模型的时间序列的这一内在特性:

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/AR1.png)

### MA移动平均模型

举个例子：

有一个生产者生产包，而且这个包随处都能买到。在一个竞争激烈地市场里，他
的包一个也没有卖出去。因此有一天，他想着改了一下包的设计，生产出一种完
全不同的包，而且这种包只有他这里能买到。第一天开卖，他的1000个包（我们
记为X(t)）全卖完了，导致有100个古怪的顾客买不到了。这100个古怪的顾客，
我们可以看做是t时刻的误差。随着时间推移，市场上越来越多同样的包，



This equation is known as AR(1) formulation. The numeral one (1) denotes that the next instance is solely dependent on the previous instance.  The alpha is a coefficient which we seek so as to minimize the error function. Notice that x(t- 1) is indeed linked to x(t-2) in the same fashion. Hence, any shock to x(t) will gradually fade off in future.

For instance, let’s say x(t) is the number of juice bottles sold in a city on a particular day. During winters, very few vendors purchased juice bottles. Suddenly, on a particular day, the temperature rose and the demand of juice bottles soared to 1000. However, after a few days, the climate became cold again. But, knowing that the people got used to drinking juice during the hot days, there were 50% of the people still drinking juice during the cold days. In following days, the proportion went down to 25% (50% of 50%) and then gradually to a small number after significant number of days. The following graph explains the inertia property of AR series:



et’s understanding AR models using the case below:

The current GDP of a country say x(t) is dependent on the last year’s GDP i.e. x(t – 1). The hypothesis being that the total cost of production of products & services in a country in a fiscal year (known as GDP) is dependent on the set up of manufacturing plants / services in the previous year and the newly set up industries / plants / services in the current year. But the primary component of the GDP is the former one.

Hence, we can formally write the equation of GDP as:
