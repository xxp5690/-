"""
业务价值分析:累计增益曲线 + Lift 提升表
回答:只给模型打分最高的 X% 用户发营销短信,能覆盖多少实际购买者?
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
DATA = os.path.join(PROJ, 'data', 'user_features.csv')
MODEL = os.path.join(PROJ, 'models', 'xgboost_double12.json')
FIG_DIR = os.path.join(PROJ, 'figures')

# ====== 复现测试集(必须与训练时相同的 random_state=42) ======
df = pd.read_csv(DATA)
df = df.replace({r'\N': np.nan, 'NULL': np.nan, '': np.nan})
df = df.drop(columns=['user_id'])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)

X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== 加载模型打分 ======
model = xgb.Booster()
model.load_model(MODEL)
y_prob = model.predict(xgb.DMatrix(X_test))

# ====== 按预测概率降序排列 ======
result = pd.DataFrame({'prob': y_prob, 'label': y_test.values})
result = result.sort_values('prob', ascending=False).reset_index(drop=True)
n = len(result)
total_buyers = result['label'].sum()
base_rate = result['label'].mean()

# ====== 分 10 组计算 Lift ======
result['decile'] = pd.qcut(result.index, 10, labels=False) + 1
lift_table = result.groupby('decile').agg(
    用户数=('label', 'count'),
    购买数=('label', 'sum'),
    购买率=('label', 'mean')
).reset_index()
lift_table['Lift提升度'] = (lift_table['购买率'] / base_rate).round(2)
lift_table['购买率'] = (lift_table['购买率'] * 100).round(1).astype(str) + '%'
print("=" * 55)
print("Lift 提升表(按模型打分从高到低分10组)")
print("=" * 55)
print(lift_table.to_string(index=False))

# ====== 关键业务结论 ======
print("\n" + "=" * 55)
print("关键业务结论")
print("=" * 55)
for top_pct in [0.1, 0.2, 0.3]:
    k = int(n * top_pct)
    captured = result.head(k)['label'].sum()
    coverage = captured / total_buyers
    top_rate = result.head(k)['label'].mean()
    lift = top_rate / base_rate
    print(f"触达 Top {int(top_pct*100):>2}% 用户({k:>4}人): "
          f"覆盖 {coverage*100:5.1f}% 的购买者, "
          f"该群体购买率 {top_rate*100:5.1f}%, Lift = {lift:.2f}")

# ====== 累计增益曲线 ======
cum_gain = result['label'].cumsum() / total_buyers
x_pct = np.arange(1, n + 1) / n

plt.figure(figsize=(8, 6))
plt.plot(x_pct, cum_gain, color='#d62728', lw=2.5, label='XGBoost 模型')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='无差别群发')

# 标注 Top 20% 点
k20 = int(n * 0.2)
cov20 = result.head(k20)['label'].sum() / total_buyers
plt.scatter([0.2], [cov20], color='blue', zorder=5, s=60)
plt.annotate(f'Top 20% 用户\n覆盖 {cov20*100:.1f}% 购买者',
             xy=(0.2, cov20), xytext=(0.32, cov20 - 0.15),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=11, color='blue')

plt.xlabel('触达用户比例', fontsize=12)
plt.ylabel('累计覆盖购买者比例', fontsize=12)
plt.title('累计增益曲线 - 双12营销人群圈选', fontsize=13)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'cumulative_gain.png'))
plt.close()
print(f"\n  -> figures/cumulative_gain.png")
