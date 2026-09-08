"""图6-1 差异化采购策略图(频率x重要性矩阵). 风格统一, 可编辑矢量图."""
import sys, io, warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for f in fm.findSystemFonts():
    try:
        if any(n in f.lower() for n in ['simhei','msyh','yahei','simsun']): fm.fontManager.addfont(f)
    except: pass
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

# 2x2 矩阵: 横轴=需求频率, 纵轴=重要性
fig, ax = plt.subplots(figsize=(8, 7))

# 四象限背景色
# 高频x高重要(右上): 暖橙浅 -> 框架协议
# 高频x低重要(右下): 蓝灰浅 -> 常规库存
# 低频x高重要(左上): 暖橙中 -> 安全库存
# 低频x低重要(左下): 蓝灰中 -> 触发式
# 坐标: 左低右高? 用 低->高 从下到上(重要性), 低->高 从左到右(频率)
# 象限矩形
# 左上(低频x高重要): x[0,5], y[5,10]
ax.add_patch(plt.Rectangle((0, 5), 5, 5, facecolor='#B86B4D', alpha=0.25, edgecolor='#2C3E50', lw=1.5))
# 右上(高频x高重要): x[5,10], y[5,10]
ax.add_patch(plt.Rectangle((5, 5), 5, 5, facecolor='#B86B4D', alpha=0.45, edgecolor='#2C3E50', lw=1.5))
# 左下(低频x低重要): x[0,5], y[0,5]
ax.add_patch(plt.Rectangle((0, 0), 5, 5, facecolor='#5B7F9A', alpha=0.25, edgecolor='#2C3E50', lw=1.5))
# 右下(高频x低重要): x[5,10], y[0,5]
ax.add_patch(plt.Rectangle((5, 0), 5, 5, facecolor='#5B7F9A', alpha=0.45, edgecolor='#2C3E50', lw=1.5))

# 四象限文字(全部调大)
# 左上: 安全库存
ax.text(2.5, 7.6, '安全库存', ha='center', va='center', fontsize=20, fontweight='bold', color='#2C3E50')
ax.text(2.5, 6.4, '低频·高重要\n(OPGW/绝缘子串)', ha='center', va='center', fontsize=13, color='#2C3E50')
ax.text(2.5, 5.3, '区间宽→战略储备\n防缺货', ha='center', va='center', fontsize=12, color='#B86B4D')
# 右上: 框架协议
ax.text(7.5, 7.6, '框架协议', ha='center', va='center', fontsize=20, fontweight='bold', color='#2C3E50')
ax.text(7.5, 6.4, '高频·高重要\n(避雷器/互感器)', ha='center', va='center', fontsize=13, color='#2C3E50')
ax.text(7.5, 5.3, '区间窄→批量备货\n稳定供应', ha='center', va='center', fontsize=12, color='#B86B4D')
# 左下: 触发式
ax.text(2.5, 2.6, '触发式采购', ha='center', va='center', fontsize=20, fontweight='bold', color='#2C3E50')
ax.text(2.5, 1.4, '低频·低重要\n(项目专用件)', ha='center', va='center', fontsize=13, color='#2C3E50')
ax.text(2.5, 0.3, '有需求信号才买\n零库存', ha='center', va='center', fontsize=12, color='#5B7F9A')
# 右下: 常规库存
ax.text(7.5, 2.6, '常规库存', ha='center', va='center', fontsize=20, fontweight='bold', color='#2C3E50')
ax.text(7.5, 1.4, '高频·低重要\n(端子箱/屏柜)', ha='center', va='center', fontsize=13, color='#2C3E50')
ax.text(7.5, 0.3, '按批次滚动备货', ha='center', va='center', fontsize=12, color='#5B7F9A')

# 坐标轴
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xticks([0, 10]); ax.set_xticklabels(['低频\n(稀疏簇)', '高频\n(主体簇)'], fontsize=15)
ax.set_yticks([0, 10]); ax.set_yticklabels(['低重要性', '高重要性'], fontsize=15)
# 坐标轴名称: 移到更靠近轴线
ax.set_xlabel('需求频率（基于 ADI 双簇）', fontsize=17, labelpad=-25)
ax.set_ylabel('物资重要性（Kraljic）', fontsize=17, labelpad=-28)
ax.tick_params(axis='both', length=0)

# 边界线(十字)
ax.axvline(5, color='#2C3E50', lw=1.5)
ax.axhline(5, color='#2C3E50', lw=1.5)

plt.tight_layout()
plt.savefig('outputs/figures/fig6-1_strategy_matrix.svg', bbox_inches='tight')
plt.savefig('outputs/figures/fig6-1_strategy_matrix.png', dpi=400, bbox_inches='tight')
print('图6-1 已保存')

import shutil, os
dstdir = r'D:\Users\dell\Desktop\研究生文献收集'
for src_n, dst_n in [('fig6-1_strategy_matrix.svg','图6-1_差异化采购策略图.svg'),
                     ('fig6-1_strategy_matrix.png','图6-1_差异化采购策略图.png')]:
    shutil.copy2(os.path.join(r'D:\Users\dell\PycharmProjects\vmd-catboost\outputs\figures', src_n),
                 os.path.join(dstdir, dst_n))
print('已复制到桌面')
