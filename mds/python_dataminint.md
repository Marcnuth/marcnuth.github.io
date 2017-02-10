###  读写
- for line in opened_file:  逐行读取，读一行加载一行
- for line in opened_file.readlines(): 逐行读取，将所有内容加载到内存
- for row in cursor: 逐row读取，读一行加载一行
- for row in cursor.fetchall() : 逐row读取，将所有内容加载到内存

### Statistics 
#### Seaborn 
> 注意: 在notebook中画图: %matplotlib inline
- 画单个变量分布图: sns.distplot:
- 画两个变量分布图+散点图: sns.jointplot
- 画多个变量分布图+散点图: sns.pairplot
> 参考: http://seaborn.pydata.org/tutorial/distributions.html


#### matplot
- acf图: matplotlib.pyplot.acorr
> 参考: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.acorr

#### Statsmodels
- acf图: statsmodels.tsa.stattools.acf
> 参考: http://statsmodels.sourceforge.net/stable/examples/notebooks/generated/tsa_arma_0.html
> 参考: http://conference.scipy.org/scipy2011/slides/mckinney_time_series.pdf


### Pandas
#### Exceptions
- UserWarning: Boolean Series key will be reindexed to match DataFrame index.
> Change df[df.time >= 0][df <= 30] to df[(df.time >=0) & (df.time <=30)]
