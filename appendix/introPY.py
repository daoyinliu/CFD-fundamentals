### 任务: 通过作图法求解方程A*exp(-E/T) = T^n
### 并将计算结果和fsolve求解结果对比
import numpy as np
from scipy.optimize import root,fsolve
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

### 解法一
def f1(x):
    return 1.0e20*np.exp(-24358/x)-x**3
Tc0 = fsolve(f1,[900])
print ('Solution is %f\n' % Tc0)

### 解法二
def f2(x):
    return 1.0e20*np.exp(-24358/x)  
def f3(x):
    return x**3

x = np.linspace(500,1000,10)
y2 = np.zeros((len(x),1))
y3 = np.zeros((len(x),1))
for i in range(len(x)):
    y2[i] = f2(x[i])
    y3[i] = f3(x[i])
    
# 图格式整体设置
plt.rcParams['font.serif'] = ['Arial']  # 
plt.figure(figsize=(6, 4))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

# 线条的记号、颜色、线型的设置
plt.plot(x, y2, marker='o', color="blue", label="y2", linewidth=1.5)
plt.plot(x, y3, marker='x', color="green", label="y3", linewidth=1.5)

# 坐标轴的设置
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel("temperature (K)", fontsize=15, fontweight='bold')
plt.ylabel("y (-)", fontsize=15, fontweight='bold')
plt.xticks(fontsize=15, fontweight='normal')  # 默认10, fontweight='bold'
plt.yticks(fontsize=15, fontweight='normal')
plt.xlim(500, 1000)
#plt.ylim(1.5, 16)

# 图例的设置
plt.legend(loc=0, numpoints=1) #显示各曲线的图例
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=15, fontweight='normal')  # 设置图例字体的大小和粗细

# 保存与显示图片
plt.savefig('./plot_demo.png', format='png', dpi=600, bbox_inches='tight')
plt.show()
