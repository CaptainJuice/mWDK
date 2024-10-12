import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results =  [

[0.489,0.419,0.503,0.518,0.535,0.523,0.511],

[0.522,0.566,0.535,0.521,0.503,0.515,0.517],

[0.468,0.587,0.576,0.573,0.562,0.549,0.554],

[0.422,0.541,0.556,0.571,0.549,0.557,0.571],

[0.363,0.477,0.508,0.519,0.519,0.522,0.511],
[0.319,0.387,0.432,0.466,0.473,0.46,0.468],
[0.263,0.354,0.359,0.358,0.361,0.387,0.382],
]
sns.set_theme()

# 创建热力图
ax = sns.heatmap(results,cmap='Blues',annot=True,fmt='.3f')

# 添加颜色条的标注
cbar = ax.collections[0].colorbar




# 添加标题和轴标签
plt.xlabel('param. $h$')
plt.ylabel('param. $\psi$')

cbar = ax.collections[0].colorbar
cbar.ax.text(0.5, 1.05, "NMI", ha='center', va='center', fontsize=12, fontweight='bold', transform=cbar.ax.transAxes)

x_labels = [1,3,5,7,9,11,13]
y_labels = [4,8,16,32,64,128,256][::-1]
x_ticks = [1,2,3,4,5,6,7]
y_ticks = [1,2,3,4,5,6,7]
ax.set_xticks([0.5, 1.5, 2.5,3.5,4.5,5.5,6.5])
ax.set_yticks([0.5, 1.5, 2.5,3.5,4.5,5.5,6.5])


ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)
# plt.xticks(ticks=x_ticks, labels=x_labels)
# plt.yticks(ticks=y_ticks, labels=y_labels)
# 显示热力图
# plt.show()

plt.savefig('sensitivity.png', transparent=True)
