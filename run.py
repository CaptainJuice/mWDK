import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
def formatnum(x, pos):
    return '$10^{%d}$' % (x)


# plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
#折线图
x = [3,4,5,6]#点的横坐标
# x=[1000,10000,100000,1000000]

formatter1 = FuncFormatter(formatnum)
formatter2 = FuncFormatter(formatnum)

f, ax = plt.subplots(1, 1)
ax.patch.set_facecolor('none')  # 设置底色为透明
f.patch.set_visible(False)
edg =  [121,13816,140923,18694195]
iik = [0.29,	3.97,	49.65,	572.13]#线1的纵坐标
wl = [0.01,0.13,   3.02,   50.56,]#线2的纵坐标
ik = [0.27, 4.01, 59.30, 680.05]
daegc = [47,121.37, 1547.22,0]
NDLS=[0.11,2.35,92.8,1642.5,]
iik = np.log10(iik)
ik = np.log10(ik)
wl = np.log10(wl)
NDLS = np.log10(NDLS)
daegc = np.log10(daegc)
plt.locator_params(axis='x', nbins=4)


plt.plot(x,wl,'s-',color = 'b',label="WL")#s-:方形
plt.plot(x,ik,'o-',color = 'g',label="WDK")#o-:圆形
plt.plot(x,iik,'o-',color = 'r',label="mWDK")#o-:圆形
plt.plot(x,daegc,'s-',color = 'y',label="DAEGC")#s-:方形
plt.plot(x,NDLS,'o-',color = 'k',label="NDLS")#o-:圆形


plt.xticks(fontproperties = 'Times New Roman', size = 13)
plt.yticks(fontproperties = 'Times New Roman', size = 13)

plt.ylabel("Time / second",size=13)#纵坐标名字
le = plt.legend(loc = "best",prop = {'size':13})#图例
ax.xaxis.set_major_formatter(formatter1)
ax.yaxis.set_major_formatter(formatter2)
plt.savefig('C:\\Users\\Admin\\Desktop\\scale_up.png', dpi = 800)

# le.get_frame().set_facecolor('none')  # 设置图例的背景为透明
le.get_frame().set_alpha(0)
plt.show()

my_dic = {'p': 'q', 'q': 'p', 'b': 'd', 'd': 'b'}

temp = 'xwmnilouv'
for t in temp:
    my_dic[t] = t
n = int(input())
for i in range(n):
    f = 0
    s = list(input())
    t = s[::-1]
    for j in range(len(s) // 2+1):
        if s[j] not in my_dic or t[j] not in my_dic or s[j] != my_dic[t[j]]:
            print('No')
            f = 1
            break
    if f == 0:
        print('Yes')



