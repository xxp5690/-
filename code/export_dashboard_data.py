"""导出 FineBI 需要的模型结果数据:Lift 表 + 特征重要性 + KPI"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
DATA = os.path.join(PROJ, 'data', 'user_features.csv')
MODEL = os.path.join(PROJ, 'models', 'xgboost_double12.json')
OUT = os.path.join(PROJ, 'data', 'dashboard')
os.makedirs(OUT, exist_ok=True)

# 复现测试集(与训练时一致的 random_state=42)
df = pd.read_csv(DATA).replace({r'\N': np.nan, 'NULL': np.nan, '': np.nan})
df = df.drop(columns=['user_id'])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)
X, y = df.drop(columns=['label']), df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.Booster()
model.load_model(MODEL)
result = pd.DataFrame({
    'prob': model.predict(xgb.DMatrix(X_test)),
    'label': y_test.values
}).sort_values('prob', ascending=False).reset_index(drop=True)
base_rate = result['label'].mean()

# 1) Lift 提升表
result['decile'] = pd.qcut(result.index, 10, labels=False) + 1
lift = result.groupby('decile').agg(
    用户数=('label', 'count'),
    购买数=('label', 'sum'),
    购买率=('label', 'mean')
).reset_index()
lift['Lift提升度'] = (lift['购买率'] / base_rate).round(2)
lift['分组'] = lift['decile'].apply(lambda x: f'Decile{x}')
lift[['分组', '用户数', '购买数', '购买率', 'Lift提升度']].to_csv(
    os.path.join(OUT, 'lift_table.csv'), index=False, encoding='utf-8-sig')

# 2) 特征重要性
imp = model.get_score(importance_type='gain')
imp_df = pd.DataFrame({'特征': list(imp.keys()), '增益': list(imp.values())})
imp_df = imp_df.sort_values('增益', ascending=False)
imp_df.to_csv(os.path.join(OUT, 'feature_importance.csv'),
              index=False, encoding='utf-8-sig')

# 3) KPI 指标卡
pd.DataFrame({
    '指标': ['AUC', '样本总数', '双12购买用户', '正样本占比', 'Top30%覆盖率', 'Top30%Lift'],
    '数值': ['0.7517', '10000', '3897', '38.97%', '49.6%', '1.65']
}).to_csv(os.path.join(OUT, 'kpi.csv'), index=False, encoding='utf-8-sig')

print('已导出 3 个 CSV 到', OUT)
for f in ['lift_table.csv', 'feature_importance.csv', 'kpi.csv']:
    print('  -', f, os.path.getsize(os.path.join(OUT, f)), '字节')