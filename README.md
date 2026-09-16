# 淘宝用户行为分析与双12大促响应预测

基于阿里天池用户行为数据集(1225 万条记录),通过 **Hadoop + Hive + XGBoost + FineBI** 构建完整的电商用户行为分析与营销响应预测项目。

## 项目亮点

- **数仓架构**:设计 ODS → DWD → ADS 三层 Hive 数仓,从原始 CSV 到汇总报表一键产出
- **漏斗分析**:浏览 → 收藏 → 加购 → 购买,全链路转化诊断
- **RFM 用户分层**:4 类用户识别,重要价值用户占 43.6%
- **XGBoost 建模**:构建 15 维用户行为特征,测试集 **AUC = 0.7517**
- **Lift 业务验证**:Top 30% 高意向用户覆盖 49.6% 实际购买者,**Lift = 1.65**
- **FineBI 可视化大屏**:7 个交互组件,覆盖 KPI 指标卡、漏斗图、双折线趋势图、RFM 饼图、特征重要性条形图、Lift 提升柱图、留存率折线

## 技术栈

| 分类 | 技术 |
|------|------|
| 大数据 | Hadoop 3.x / Hive 3.1.3 / HDFS / YARN |
| 机器学习 | XGBoost 3.4 / scikit-learn |
| 数据处理 | Python 3.13 / pandas / numpy |
| 可视化 | FineBI 6.0 / matplotlib |
| 集群 | 3 节点伪分布式(VirtualBox,4GB/节点) |

## 数据流向

```
天池原始 CSV(508MB)
    ↓ HDFS /ecommerce/raw/
ODS: user_behavior_raw (外部表)
    ↓ 清洗去重 + ORC 压缩
DWD: dwd_user_behavior (1225万行,分区)
    ↓ SQL 聚合
ADS: ads_funnel / ads_daily_trend / ads_retention / ads_rfm_segment
    ↓ HiveServer2 JDBC
FineBI 6.0 大屏(抽取模式,秒级响应)

DWD → Python 特征工程 → user_features(10000行)
    ↓ XGBoost
    ├─ 模型评估(ROC/AUC/混淆矩阵)
    ├─ Lift 业务分析
    └─ FineBI 特征重要性 / Lift 提升组件
```

## 关键结果

### 1. 用户行为漏斗

| 阶段 | 用户数 | 转化率 |
|------|--------|--------|
| 浏览 | 10,000 | 100% |
| 收藏 | 6,730 | 67.3% |
| 加购 | 8,614 | 86.1% |
| 购买 | 8,886 | 88.9% |

### 2. XGBoost 模型

| 指标 | 值 |
|------|------|
| 测试集 AUC | **0.7517** |
| Accuracy | 0.6975 |
| Precision(购买) | 0.6246 |
| Recall(购买) | 0.5597 |

**Top 5 重要特征**:
1. `preheat_3d_cnt`(大促前 3 天预热行为,Gain ≈ 22)
2. `active_days_obs`(活跃天数)
3. `cart_cnt_obs`(加购数)
4. `buy_cnt_obs`(购买数)
5. `buy_view_rate`(购买浏览转化率)

### 3. Lift 业务价值

| 触达人群 | 覆盖购买者比例 | 该群体购买率 | Lift |
|----------|---------------|-------------|------|
| Top 10% | 18.7% | 73.0% | 1.87 |
| Top 20% | 36.1% | 70.2% | 1.80 |
| Top 30% | **49.6%** | **64.3%** | **1.65** |

> 同等营销预算下,用模型圈选 Top 30% 用户群发,比无差别触达效率提升 **65%**。

### 4. RFM 用户分层

| 用户类型 | 用户数 | 占比 |
|----------|--------|------|
| 重要价值用户(高R高F) | 3,873 | 43.6% |
| 重要挽留用户(低R低F) | 2,097 | 23.6% |
| 重要发展用户(高R低F) | 1,458 | 16.4% |
| 重要保持用户(低R高F) | 1,458 | 16.4% |

## 项目结构

```
淘宝用户行为分析/
├── code/
│   ├── train_xgboost.py          # XGBoost 训练 + 评估 + 可视化
│   ├── business_analysis.py      # Lift 业务分析 + 累计增益曲线
│   └── export_dashboard_data.py  # 导出 FineBI 所需 CSV
├── figures/                       # matplotlib 静态图
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── cumulative_gain.png
├── models/
│   └── xgboost_double12.json     # 训练好的模型
├── docs/
│   └── 仪表板.pdf                # FineBI 大屏导出 PDF
├── .gitignore
└── README.md
```

## 复现步骤

### 1. 准备环境

- 3 节点 Hadoop 集群(4GB/节点)
- Hive 3.x(MySQL metastore + HiveServer2)
- Python 3.13 + 依赖:
  ```bash
  pip install xgboost pandas numpy scikit-learn matplotlib
  ```
- FineBI 6.0(可选,用于可视化大屏)

### 2. Hive 侧(在 hadoop100 上)

```sql
-- 建数据库
CREATE DATABASE ecommerce;
USE ecommerce;

-- ① 建外部表指向原始 CSV(先上传到 HDFS /ecommerce/raw)
CREATE EXTERNAL TABLE user_behavior_raw (
    user_id STRING,
    item_id STRING,
    item_category STRING,
    behavior_type INT,
    event_time STRING
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/ecommerce/raw';

-- ② 建 DWD 清洗表(ORC 压缩,分区)
CREATE TABLE dwd_user_behavior (
    user_id STRING,
    item_id STRING,
    item_category STRING,
    behavior_type INT,
    event_time STRING,
    event_date STRING,
    event_hour INT
) STORED AS ORC;

INSERT OVERWRITE TABLE dwd_user_behavior
SELECT user_id, item_id, item_category, behavior_type,
       event_time,
       SUBSTR(event_time,1,10) AS event_date,
       CAST(SUBSTR(event_time,12,2) AS INT) AS event_hour
FROM user_behavior_raw
WHERE event_time IS NOT NULL AND user_id IS NOT NULL;

-- ③ ADS 汇总表(大屏数据源)
CREATE TABLE ads_funnel AS ...;         -- 4 行漏斗
CREATE TABLE ads_daily_trend AS ...;    -- 31 行日活趋势
CREATE TABLE ads_retention AS ...;      -- 31 行留存
CREATE TABLE ads_rfm_segment AS ...;    -- 4 行 RFM 分层
```

### 3. Python 侧(在本地)

```bash
# 先从 Hive 导出 user_features.csv(或跑特征工程 SQL)
# 再运行训练
python code/train_xgboost.py
python code/business_analysis.py
python code/export_dashboard_data.py
```

### 4. FineBI 侧

1. 配置 HiveServer2 数据连接:`jdbc:hive2://hadoop100:10000/ecommerce`
2. 抽取 4 张 ADS 表 + 上传 3 个 CSV(lift_table / feature_importance / kpi)
3. 新建仪表板,7 个组件拖拽配置(详见 docs/仪表板.pdf)

## 数据集说明

本项目使用的原始数据集来自**阿里天池"阿里移动推荐算法"比赛**,请勿将原始数据上传至任何公共仓库。项目中 `data/` 目录默认通过 `.gitignore` 排除。

## 简历话术

> **淘宝用户行为分析与双12大促响应预测**
> - 基于 Hadoop+Hive 搭建 3 节点集群,处理天池 1225 万条用户行为数据,设计 ODS→DWD→ADS 三层数仓架构
> - 使用 SQL 完成漏斗转化(购买率 88.86%)、RFM 用户分层(重要价值用户 43.6%)、留存率分析
> - 构建 15 维用户行为特征,XGBoost 预测双12购买行为,测试集 AUC 0.7517
> - 通过 Lift 分析验证:Top 30% 高意向用户覆盖 49.6% 实际购买者,Lift=1.65,同等预算下营销效率提升 65%
> - 使用 FineBI 6.0 连接 HiveServer2 搭建交互式可视化大屏,含 7 个分析组件
> - **技术栈**:Hadoop / Hive / XGBoost / Python / FineBI
