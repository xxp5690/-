"""
淘宝用户行为数据分析 - 双12大促响应预测
XGBoost 二分类模型训练 + 评估 + 可视化
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# ====== 中文显示 ======
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ====== 路径配置(相对脚本位置,自动定位) ======
BASE = os.path.dirname(os.path.abspath(__file__))   # code/
PROJ = os.path.dirname(BASE)                          # 项目根目录
DATA = os.path.join(PROJ, 'data', 'user_features.csv')
FIG_DIR = os.path.join(PROJ, 'figures')
MODEL_DIR = os.path.join(PROJ, 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_check():
    """加载并校验数据"""
    print("=" * 60)
    print("步骤1: 数据加载与校验")
    print("=" * 60)
    df = pd.read_csv(DATA)
    print(f"数据形状: {df.shape}  (期望: 10000, 17)")
    print(f"\n标签分布:")
    print(df['label'].value_counts(normalize=True).round(4))
    print(f"\n缺失值统计:")
    null_cnt = df.isnull().sum()
    print(null_cnt[null_cnt > 0] if null_cnt.sum() > 0 else "无缺失值")
    print(f"\n前3行预览:")
    print(df.head(3))
    return df

def preprocess(df):
    """数据预处理"""
    print("\n" + "=" * 60)
    print("步骤2: 数据预处理")
    print("=" * 60)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
        print("已移除 user_id 列")

    # Hive 导出时 NULL 会变成 '\N' 字符串,先转成真正的 NaN
    df = df.replace({r'\N': np.nan, 'NULL': np.nan, '': np.nan})

    # 所有特征列强制转为数值类型(无法转换的变 NaN)
    feature_cols = [c for c in df.columns if c != 'label']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 填充 NaN 为 0
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        print(f"填充 NULL 列: {null_cols}")
        df[null_cols] = df[null_cols].fillna(0)
    else:
        print("无 NULL,无需填充")

    # 打印各列类型,确认全部是数值
    print(f"\n字段类型检查:")
    print(df.dtypes)

    X = df.drop(columns=['label'])
    y = df['label']
    print(f"特征数: {X.shape[1]}")
    print(f"正样本: {(y == 1).sum()}, 负样本: {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape[0]} (正样本 {y_train.sum()})")
    print(f"测试集: {X_test.shape[0]} (正样本 {y_test.sum()})")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """训练 XGBoost"""
    print("\n" + "=" * 60)
    print("步骤3: 训练 XGBoost 模型")
    print("=" * 60)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'lambda': 1.0,
        'seed': 42,
        'verbosity': 1
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(
        params, dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train')],
        early_stopping_rounds=20,
        verbose_eval=20
    )
    print(f"\n最佳迭代轮次: {model.best_iteration}")
    return model

def evaluate(model, X_test, y_test):
    """评估模型"""
    print("\n" + "=" * 60)
    print("步骤4: 模型评估")
    print("=" * 60)
    dtest = xgb.DMatrix(X_test)
    y_prob = model.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"AUC:        {auc:.4f}  (目标 0.72-0.82)")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred))
    return y_prob, y_pred

def plot_results(model, X_test, y_test, y_prob, y_pred):
    """可视化"""
    print("\n" + "=" * 60)
    print("步骤5: 可视化输出")
    print("=" * 60)

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#d62728', lw=2,
             label=f'XGBoost (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='随机猜测')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('假正率 (FPR)', fontsize=12)
    plt.ylabel('真正率 (TPR)', fontsize=12)
    plt.title('ROC 曲线 - 双12大促响应预测', fontsize=13)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'roc_curve.png'))
    plt.close()
    print("  -> figures/roc_curve.png")

    # 特征重要性
    importance = model.get_score(importance_type='gain')
    imp_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'gain': list(importance.values())
    }).sort_values('gain', ascending=True)

    plt.figure(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(imp_df)))
    plt.barh(imp_df['feature'], imp_df['gain'], color=colors)
    plt.xlabel('信息增益 (Gain)', fontsize=12)
    plt.title('特征重要性 - XGBoost', fontsize=13)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'))
    plt.close()
    print("  -> figures/feature_importance.png")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=14)
    plt.xticks([0, 1], ['未购买', '购买'])
    plt.yticks([0, 1], ['未购买', '购买'])
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('真实值', fontsize=12)
    plt.title('混淆矩阵', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'))
    plt.close()
    print("  -> figures/confusion_matrix.png")

def save_model(model):
    """保存模型"""
    model_path = os.path.join(MODEL_DIR, 'xgboost_double12.json')
    model.save_model(model_path)
    print(f"\n模型已保存: {model_path}")

if __name__ == '__main__':
    df = load_and_check()
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train_model(X_train, y_train)
    y_prob, y_pred = evaluate(model, X_test, y_test)
    plot_results(model, X_test, y_test, y_prob, y_pred)
    save_model(model)
    print("\n" + "=" * 60)
    print("全部完成!图表和模型已生成")
    print("=" * 60)