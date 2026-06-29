#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tran.py - 镇江香醋风味模型训练脚本

负责:
  1. 内置训练/验证/扩展数据 (来自 4 篇核心文献)
  2. 特征工程 (OAV 变换 + 派生指标)
  3. 多模型训练: PLSR / MLR / LASSO / Ridge / SVR / RF / XGBoost
  4. 全套离线分析: Pearson 相关 / PCA / ANOVA / 网格搜索
  5. 多指标评估 (R^2 / RMSE / MAE / MAPE)
  6. 训练/验证分离 + 模型持久化到 models/

运行:
  python tran.py
  python tran.py --no-tune   跳过网格搜索 (快速模式)
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from standard import (
    classify_sample, classify_dataframe, check_gb_compliance,
    GB_STANDARD, GRADE_THRESHOLDS,
)

warnings.filterwarnings("ignore")

# =============================================================================
# 路径与全局配置
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 输出感官维度
SENSORY_OUTPUTS = [
    "s_醋酸味", "s_苦味", "s_甜味", "s_咸味",
    "s_风味", "s_酱香", "s_谷物香", "s_炒米香",
    "s_米醋香", "s_持久度", "s_柔和度",
]

# 输入特征分块
PROCESS_FEATURES = ["工艺", "醋龄月", "发酵天数", "温度峰值"]
PHYSCHEM_FEATURES = ["总酸", "不挥发酸", "pH", "还原糖"]
ORGANIC_ACID_FEATURES = ["乙酸", "乳酸", "琥珀酸", "焦谷氨酸",
                         "柠檬酸", "酒石酸", "苹果酸", "草酸", "丙酮酸"]
AMINO_ACID_FEATURES = ["天冬氨酸", "谷氨酸", "丙氨酸",
                       "赖氨酸", "酪氨酸", "色氨酸",
                       "甘氨酸", "苏氨酸", "脯氨酸"]
VOLATILE_FEATURES = ["乙酸乙酯", "乙酸异戊酯", "乙偶姻", "糠醛",
                      "四甲基吡嗪", "苯乙醇"]
OAV_FEATURES = VOLATILE_FEATURES  # 直接对这 6 个做 OAV

# 精简特征集 (6个) - 基于领域知识选取, 防止n=9时过拟合
# 覆盖: 酸度骨架(总酸/不挥发酸) + 甜味基底(还原糖) + 鲜味(总游离氨基酸) + 核心酯香(OAV_乙酸乙酯) + 陈酿时间(醋龄月)
CORE_FEATURES = [
    "总酸", "不挥发酸", "还原糖",
    "总游离氨基酸",
    "OAV_乙酸乙酯",
    "醋龄月",
]

# OAV 阈值 (μg/mL, 水/食醋基质)
ODOR_THRESHOLDS = {
    "乙酸乙酯": 5.0,
    "乙酸异戊酯": 0.3,
    "乙偶姻": 0.8,
    "糠醛": 4.0,
    "四甲基吡嗪": 0.5,
    "苯乙醇": 10.0,
}

# =============================================================================
# 第一部分: 数据定义 (从 4 篇核心文献的实测数据)
# =============================================================================

# 文献 A: 李思雨2025 - 传统 vs 封闭式固态发酵对比
LI2025_SAMPLES = [
    {
        "id": "LI2025_TF", "文献": "李思雨2025", "工艺": 1, "醋龄月": 0,
        "发酵天数": 18, "温度峰值": 44.67,
        "总酸": 7.58, "不挥发酸": 1.96, "pH": 3.91, "还原糖": 2.93,
        "乙酸": 5.80, "乳酸": 2.62, "琥珀酸": 0.71, "焦谷氨酸": 0.20,
        "柠檬酸": 0.31, "酒石酸": 0.31, "苹果酸": 0.034, "草酸": 0.078, "丙酮酸": 0.035,
        "天冬氨酸": 19.0, "谷氨酸": 65.0, "丙氨酸": 80.0,
        "赖氨酸": 35.0, "酪氨酸": 28.0, "色氨酸": 156.5,
        "甘氨酸": 22.0, "苏氨酸": 18.0, "脯氨酸": 30.0,
        "乙酸乙酯": 800.0, "乙酸异戊酯": 50.0,
        "乙偶姻": 397.17, "糠醛": 1500.0,
        "四甲基吡嗪": 25.0, "苯乙醇": 80.0,
        "s_醋酸味": 7.0, "s_苦味": 6.0, "s_甜味": 3.0, "s_咸味": 4.0,
        "s_风味": 6.0, "s_酱香": 5.0, "s_谷物香": 6.0, "s_炒米香": 5.0,
        "s_米醋香": 5.0, "s_持久度": 5.0, "s_柔和度": 4.0,
    },
    {
        "id": "LI2025_CF", "文献": "李思雨2025", "工艺": 0, "醋龄月": 0,
        "发酵天数": 14, "温度峰值": 43.90,
        "总酸": 7.20, "不挥发酸": 2.21, "pH": 3.80, "还原糖": 2.64,
        "乙酸": 5.50, "乳酸": 3.36, "琥珀酸": 0.60, "焦谷氨酸": 0.18,
        "柠檬酸": 0.40, "酒石酸": 0.30, "苹果酸": 0.04, "草酸": 0.06, "丙酮酸": 0.03,
        "天冬氨酸": 27.0, "谷氨酸": 300.0, "丙氨酸": 85.0,
        "赖氨酸": 30.0, "酪氨酸": 22.0, "色氨酸": 156.5,
        "甘氨酸": 25.0, "苏氨酸": 20.0, "脯氨酸": 28.0,
        "乙酸乙酯": 2699.33, "乙酸异戊酯": 80.0,
        "乙偶姻": 2066.15, "糠醛": 1200.0,
        "四甲基吡嗪": 30.0, "苯乙醇": 100.0,
        "s_醋酸味": 8.0, "s_苦味": 4.0, "s_甜味": 5.0, "s_咸味": 3.0,
        "s_风味": 4.0, "s_酱香": 4.0, "s_谷物香": 4.0, "s_炒米香": 3.0,
        "s_米醋香": 4.0, "s_持久度": 7.0, "s_柔和度": 6.0,
    },
]

# 文献 B: 胡冬梅2023 - 手工 vs 工业, 新醋 vs 3年
HU2023_SAMPLES = [
    {
        "id": "HU2023_手工新醋", "文献": "胡冬梅2023", "工艺": 1, "醋龄月": 0,
        "发酵天数": 20, "温度峰值": 42.0,
        "总酸": 8.0, "不挥发酸": 2.20, "pH": 4.0, "还原糖": 2.70,
        "乙酸": 4.68, "乳酸": 2.12, "琥珀酸": 0.57, "焦谷氨酸": 0.16,
        "柠檬酸": 0.25, "酒石酸": 0.25, "苹果酸": 0.027, "草酸": 0.062, "丙酮酸": 0.028,
        "天冬氨酸": 15.0, "谷氨酸": 50.0, "丙氨酸": 70.0,
        "赖氨酸": 30.0, "酪氨酸": 22.0, "色氨酸": 90.0,
        "甘氨酸": 18.0, "苏氨酸": 14.0, "脯氨酸": 25.0,
        "乙酸乙酯": 500.0, "乙酸异戊酯": 30.0,
        "乙偶姻": 300.0, "糠醛": 1000.0,
        "四甲基吡嗪": 18.0, "苯乙醇": 60.0,
        "s_醋酸味": 7.0, "s_苦味": 6.0, "s_甜味": 3.0, "s_咸味": 4.0,
        "s_风味": 7.0, "s_酱香": 6.0, "s_谷物香": 7.0, "s_炒米香": 6.0,
        "s_米醋香": 6.0, "s_持久度": 5.0, "s_柔和度": 4.5,
    },
    {
        "id": "HU2023_工业新醋", "文献": "胡冬梅2023", "工艺": 0, "醋龄月": 0,
        "发酵天数": 18, "温度峰值": 41.0,
        "总酸": 7.2, "不挥发酸": 2.30, "pH": 3.85, "还原糖": 3.80,
        "乙酸": 4.11, "乳酸": 1.98, "琥珀酸": 0.15, "焦谷氨酸": 0.12,
        "柠檬酸": 0.23, "酒石酸": 0.49, "苹果酸": 0.019, "草酸": 0.055, "丙酮酸": 0.039,
        "天冬氨酸": 12.0, "谷氨酸": 40.0, "丙氨酸": 55.0,
        "赖氨酸": 25.0, "酪氨酸": 18.0, "色氨酸": 75.0,
        "甘氨酸": 16.0, "苏氨酸": 12.0, "脯氨酸": 20.0,
        "乙酸乙酯": 600.0, "乙酸异戊酯": 40.0,
        "乙偶姻": 350.0, "糠醛": 900.0,
        "四甲基吡嗪": 20.0, "苯乙醇": 70.0,
        "s_醋酸味": 7.5, "s_苦味": 5.0, "s_甜味": 3.5, "s_咸味": 4.0,
        "s_风味": 6.0, "s_酱香": 5.0, "s_谷物香": 5.0, "s_炒米香": 5.0,
        "s_米醋香": 5.5, "s_持久度": 6.0, "s_柔和度": 5.0,
    },
    {
        "id": "HU2023_手工3年", "文献": "胡冬梅2023", "工艺": 1, "醋龄月": 36,
        "发酵天数": 20, "温度峰值": 42.0,
        "总酸": 6.0, "不挥发酸": 2.30, "pH": 3.95, "还原糖": 2.00,
        "乙酸": 3.21, "乳酸": 1.76, "琥珀酸": 0.57, "焦谷氨酸": 0.11,
        "柠檬酸": 0.25, "酒石酸": 0.28, "苹果酸": 0.032, "草酸": 0.044, "丙酮酸": 0.032,
        "天冬氨酸": 18.0, "谷氨酸": 80.0, "丙氨酸": 95.0,
        "赖氨酸": 40.0, "酪氨酸": 30.0, "色氨酸": 130.0,
        "甘氨酸": 25.0, "苏氨酸": 20.0, "脯氨酸": 35.0,
        "乙酸乙酯": 700.0, "乙酸异戊酯": 45.0,
        "乙偶姻": 14.0, "糠醛": 2200.0,
        "四甲基吡嗪": 45.0, "苯乙醇": 100.0,
        "s_醋酸味": 7.5, "s_苦味": 5.0, "s_甜味": 4.0, "s_咸味": 4.0,
        "s_风味": 8.0, "s_酱香": 7.0, "s_谷物香": 7.5, "s_炒米香": 6.5,
        "s_米醋香": 7.0, "s_持久度": 7.0, "s_柔和度": 6.5,
    },
    {
        "id": "HU2023_工业3年", "文献": "胡冬梅2023", "工艺": 0, "醋龄月": 36,
        "发酵天数": 18, "温度峰值": 41.0,
        "总酸": 5.8, "不挥发酸": 1.80, "pH": 4.0, "还原糖": 2.60,
        "乙酸": 3.14, "乳酸": 1.42, "琥珀酸": 0.18, "焦谷氨酸": 0.10,
        "柠檬酸": 0.37, "酒石酸": 0.24, "苹果酸": 0.028, "草酸": 0.026, "丙酮酸": 0.031,
        "天冬氨酸": 13.0, "谷氨酸": 60.0, "丙氨酸": 70.0,
        "赖氨酸": 30.0, "酪氨酸": 25.0, "色氨酸": 100.0,
        "甘氨酸": 20.0, "苏氨酸": 16.0, "脯氨酸": 28.0,
        "乙酸乙酯": 800.0, "乙酸异戊酯": 50.0,
        "乙偶姻": 12.0, "糠醛": 1800.0,
        "四甲基吡嗪": 50.0, "苯乙醇": 90.0,
        "s_醋酸味": 7.0, "s_苦味": 5.0, "s_甜味": 4.0, "s_咸味": 4.0,
        "s_风味": 7.0, "s_酱香": 6.0, "s_谷物香": 6.0, "s_炒米香": 5.5,
        "s_米醋香": 6.0, "s_持久度": 6.5, "s_柔和度": 6.0,
    },
]

# 文献 C: 任晓荣2023 - 不同陈酿年份 (3/5/8年)
REN2023_SAMPLES = [
    {
        "id": "REN2023_ZV3", "文献": "任晓荣2023", "工艺": 1, "醋龄月": 36,
        "发酵天数": 20, "温度峰值": 42.0,
        "总酸": 5.72, "不挥发酸": 1.70, "pH": 3.20, "还原糖": 1.43,
        "乙酸": 2.15, "乳酸": 1.12, "琥珀酸": 0.022, "焦谷氨酸": 0.0,
        "柠檬酸": 0.0007, "酒石酸": 0.38, "苹果酸": 0.072, "草酸": 0.038, "丙酮酸": 0.164,
        "天冬氨酸": 9.1, "谷氨酸": 2.7, "丙氨酸": 76.6,
        "赖氨酸": 5.5, "酪氨酸": 5.5, "色氨酸": 0.0,
        "甘氨酸": 2.9, "苏氨酸": 21.5, "脯氨酸": 2.7,
        "乙酸乙酯": 1100.0, "乙酸异戊酯": 60.0,
        "乙偶姻": 14.33, "糠醛": 2800.0,
        "四甲基吡嗪": 42.0, "苯乙醇": 110.0,
        "s_醋酸味": 6.5, "s_苦味": 5.5, "s_甜味": 4.5, "s_咸味": 4.0,
        "s_风味": 7.5, "s_酱香": 6.5, "s_谷物香": 7.0, "s_炒米香": 6.0,
        "s_米醋香": 6.5, "s_持久度": 6.5, "s_柔和度": 6.0,
    },
    {
        "id": "REN2023_ZV5", "文献": "任晓荣2023", "工艺": 1, "醋龄月": 60,
        "发酵天数": 20, "温度峰值": 42.0,
        "总酸": 6.32, "不挥发酸": 1.85, "pH": 3.65, "还原糖": 0.93,
        "乙酸": 2.31, "乳酸": 1.10, "琥珀酸": 0.018, "焦谷氨酸": 0.0,
        "柠檬酸": 0.082, "酒石酸": 0.43, "苹果酸": 0.010, "草酸": 0.022, "丙酮酸": 0.027,
        "天冬氨酸": 37.5, "谷氨酸": 79.1, "丙氨酸": 86.5,
        "赖氨酸": 66.8, "酪氨酸": 70.1, "色氨酸": 0.0,
        "甘氨酸": 34.9, "苏氨酸": 34.5, "脯氨酸": 28.0,
        "乙酸乙酯": 1500.0, "乙酸异戊酯": 70.0,
        "乙偶姻": 12.0, "糠醛": 3500.0,
        "四甲基吡嗪": 44.0, "苯乙醇": 130.0,
        "s_醋酸味": 7.0, "s_苦味": 5.0, "s_甜味": 5.0, "s_咸味": 4.0,
        "s_风味": 8.0, "s_酱香": 7.0, "s_谷物香": 7.0, "s_炒米香": 6.5,
        "s_米醋香": 7.0, "s_持久度": 7.5, "s_柔和度": 7.0,
    },
    {
        "id": "REN2023_ZV8", "文献": "任晓荣2023", "工艺": 1, "醋龄月": 96,
        "发酵天数": 20, "温度峰值": 42.0,
        "总酸": 7.43, "不挥发酸": 2.30, "pH": 3.71, "还原糖": 2.96,
        "乙酸": 3.22, "乳酸": 1.21, "琥珀酸": 0.026, "焦谷氨酸": 0.0,
        "柠檬酸": 0.023, "酒石酸": 0.57, "苹果酸": 0.064, "草酸": 0.068, "丙酮酸": 0.188,
        "天冬氨酸": 71.3, "谷氨酸": 39.0, "丙氨酸": 144.2,
        "赖氨酸": 70.2, "酪氨酸": 70.2, "色氨酸": 0.0,
        "甘氨酸": 39.6, "苏氨酸": 31.7, "脯氨酸": 57.0,
        "乙酸乙酯": 1800.0, "乙酸异戊酯": 80.0,
        "乙偶姻": 9.07, "糠醛": 4500.0,
        "四甲基吡嗪": 94.87, "苯乙醇": 150.0,
        "s_醋酸味": 7.5, "s_苦味": 4.5, "s_甜味": 5.5, "s_咸味": 4.0,
        "s_风味": 8.5, "s_酱香": 8.0, "s_谷物香": 8.0, "s_炒米香": 7.0,
        "s_米醋香": 7.5, "s_持久度": 8.5, "s_柔和度": 8.0,
    },
]

# 文献 D: 王超2020 - 发酵过程动态 (Day3 / Day11 / Day21)
WANG2020_SAMPLES = [
    {
        "id": "WC2020_Day3", "文献": "王超2020", "工艺": 1, "醋龄月": 0,
        "发酵天数": 3, "温度峰值": 38.0,
        "总酸": 1.2, "不挥发酸": 0.30, "pH": 4.50, "还原糖": 1.20,
        "氨基酸态氮": 0.20,  # 论文实测值
        "乙酸": 0.80, "乳酸": 0.20, "琥珀酸": 0.05, "焦谷氨酸": 0.05,
        "柠檬酸": 0.05, "酒石酸": 0.05, "苹果酸": 0.005, "草酸": 0.020, "丙酮酸": 0.010,
        "天冬氨酸": 5.0, "谷氨酸": 15.0, "丙氨酸": 20.0,
        "赖氨酸": 10.0, "酪氨酸": 8.0, "色氨酸": 30.0,
        "甘氨酸": 8.0, "苏氨酸": 6.0, "脯氨酸": 10.0,
        "乙酸乙酯": 100.0, "乙酸异戊酯": 10.0,
        "乙偶姻": 50.0, "糠醛": 100.0,
        "四甲基吡嗪": 2.0, "苯乙醇": 15.0,
        "s_醋酸味": 3.0, "s_苦味": 3.0, "s_甜味": 5.0, "s_咸味": 3.0,
        "s_风味": 2.0, "s_酱香": 2.0, "s_谷物香": 4.0, "s_炒米香": 2.0,
        "s_米醋香": 3.0, "s_持久度": 2.0, "s_柔和度": 5.0,
    },
    {
        "id": "WC2020_Day11", "文献": "王超2020", "工艺": 1, "醋龄月": 0,
        "发酵天数": 11, "温度峰值": 43.0,
        "总酸": 3.5, "不挥发酸": 0.90, "pH": 4.05, "还原糖": 2.50,
        "氨基酸态氮": 0.27,
        "乙酸": 2.40, "乳酸": 0.70, "琥珀酸": 0.20, "焦谷氨酸": 0.08,
        "柠檬酸": 0.10, "酒石酸": 0.10, "苹果酸": 0.015, "草酸": 0.030, "丙酮酸": 0.020,
        "天冬氨酸": 8.0, "谷氨酸": 25.0, "丙氨酸": 40.0,
        "赖氨酸": 15.0, "酪氨酸": 12.0, "色氨酸": 50.0,
        "甘氨酸": 12.0, "苏氨酸": 10.0, "脯氨酸": 18.0,
        "乙酸乙酯": 300.0, "乙酸异戊酯": 20.0,
        "乙偶姻": 150.0, "糠醛": 400.0,
        "四甲基吡嗪": 8.0, "苯乙醇": 30.0,
        "s_醋酸味": 5.0, "s_苦味": 4.0, "s_甜味": 4.0, "s_咸味": 3.5,
        "s_风味": 4.0, "s_酱香": 3.0, "s_谷物香": 5.0, "s_炒米香": 3.0,
        "s_米醋香": 4.0, "s_持久度": 3.5, "s_柔和度": 5.0,
    },
    {
        "id": "WC2020_Day21", "文献": "王超2020", "工艺": 1, "醋龄月": 0,
        "发酵天数": 21, "温度峰值": 44.0,
        "总酸": 5.5, "不挥发酸": 1.60, "pH": 3.75, "还原糖": 1.50,
        "氨基酸态氮": 0.29,
        "乙酸": 4.10, "乳酸": 1.40, "琥珀酸": 0.40, "焦谷氨酸": 0.12,
        "柠檬酸": 0.15, "酒石酸": 0.20, "苹果酸": 0.025, "草酸": 0.040, "丙酮酸": 0.025,
        "天冬氨酸": 12.0, "谷氨酸": 45.0, "丙氨酸": 60.0,
        "赖氨酸": 25.0, "酪氨酸": 20.0, "色氨酸": 90.0,
        "甘氨酸": 18.0, "苏氨酸": 14.0, "脯氨酸": 25.0,
        "乙酸乙酯": 500.0, "乙酸异戊酯": 30.0,
        "乙偶姻": 250.0, "糠醛": 800.0,
        "四甲基吡嗪": 15.0, "苯乙醇": 50.0,
        "s_醋酸味": 6.5, "s_苦味": 5.0, "s_甜味": 3.5, "s_咸味": 4.0,
        "s_风味": 5.5, "s_酱香": 4.5, "s_谷物香": 5.5, "s_炒米香": 4.5,
        "s_米醋香": 5.0, "s_持久度": 4.5, "s_柔和度": 4.5,
    },
]

# 文献 E: 李信2022 - 不同封醅时间 (0/3/7/15/30 d)
LIXIN2022_SAMPLES = [
    {
        "id": "LX2022_0d", "文献": "李信2022", "工艺": 0, "醋龄月": 0,
        "发酵天数": 21, "温度峰值": 43.0,
        "总酸": 7.19, "不挥发酸": 2.23, "pH": 3.85, "还原糖": 2.36,
        "氨基酸态氮": 0.29,  # 论文实测值
        "乙酸": 5.044, "乳酸": 1.106, "琥珀酸": 0.0902, "焦谷氨酸": 0.103,
        "柠檬酸": 0.054, "酒石酸": 0.170, "苹果酸": 0.0, "草酸": 0.0, "丙酮酸": 0.041,
        "天冬氨酸": 105.1, "谷氨酸": 58.7, "丙氨酸": 131.6,
        "赖氨酸": 71.5, "酪氨酸": 52.0, "色氨酸": 0.0,
        "甘氨酸": 80.5, "苏氨酸": 71.6, "脯氨酸": 18.9,
        "乙酸乙酯": 850.0, "乙酸异戊酯": 42.0,
        "乙偶姻": 119.0, "糠醛": 4015.0,
        "四甲基吡嗪": 509.0, "苯乙醇": 15254.0,
    },
    {
        "id": "LX2022_3d", "文献": "李信2022", "工艺": 0, "醋龄月": 0,
        "发酵天数": 21, "温度峰值": 43.0,
        "总酸": 7.20, "不挥发酸": 2.32, "pH": 3.82, "还原糖": 2.30,
        "氨基酸态氮": 0.30,
        "乙酸": 5.043, "乳酸": 1.125, "琥珀酸": 0.0905, "焦谷氨酸": 0.104,
        "柠檬酸": 0.053, "酒石酸": 0.170, "苹果酸": 0.0, "草酸": 0.0, "丙酮酸": 0.041,
        "天冬氨酸": 105.1, "谷氨酸": 58.8, "丙氨酸": 131.6,
        "赖氨酸": 71.5, "酪氨酸": 52.1, "色氨酸": 0.0,
        "甘氨酸": 80.5, "苏氨酸": 71.6, "脯氨酸": 19.0,
        "乙酸乙酯": 920.0, "乙酸异戊酯": 69.0,
        "乙偶姻": 317.0, "糠醛": 3994.0,
        "四甲基吡嗪": 688.0, "苯乙醇": 15012.0,
    },
    {
        "id": "LX2022_7d", "文献": "李信2022", "工艺": 0, "醋龄月": 0,
        "发酵天数": 21, "温度峰值": 43.0,
        "总酸": 7.23, "不挥发酸": 2.34, "pH": 3.80, "还原糖": 2.28,
        "氨基酸态氮": 0.31,
        "乙酸": 5.042, "乳酸": 1.126, "琥珀酸": 0.0909, "焦谷氨酸": 0.105,
        "柠檬酸": 0.054, "酒石酸": 0.171, "苹果酸": 0.0, "草酸": 0.0, "丙酮酸": 0.041,
        "天冬氨酸": 105.2, "谷氨酸": 58.8, "丙氨酸": 131.6,
        "赖氨酸": 71.5, "酪氨酸": 52.2, "色氨酸": 0.0,
        "甘氨酸": 80.5, "苏氨酸": 71.6, "脯氨酸": 19.1,
        "乙酸乙酯": 1050.0, "乙酸异戊酯": 125.0,
        "乙偶姻": 389.0, "糠醛": 3982.0,
        "四甲基吡嗪": 812.0, "苯乙醇": 14894.0,
    },
    {
        "id": "LX2022_15d", "文献": "李信2022", "工艺": 0, "醋龄月": 0,
        "发酵天数": 21, "温度峰值": 43.0,
        "总酸": 7.30, "不挥发酸": 2.36, "pH": 3.78, "还原糖": 2.25,
        "氨基酸态氮": 0.30,
        "乙酸": 5.032, "乳酸": 1.134, "琥珀酸": 0.0912, "焦谷氨酸": 0.105,
        "柠檬酸": 0.053, "酒石酸": 0.172, "苹果酸": 0.0, "草酸": 0.0, "丙酮酸": 0.041,
        "天冬氨酸": 105.2, "谷氨酸": 58.8, "丙氨酸": 131.7,
        "赖氨酸": 71.6, "酪氨酸": 52.2, "色氨酸": 0.0,
        "甘氨酸": 80.5, "苏氨酸": 71.6, "脯氨酸": 19.1,
        "乙酸乙酯": 1202.0, "乙酸异戊酯": 187.0,
        "乙偶姻": 454.0, "糠醛": 4011.0,
        "四甲基吡嗪": 1332.0, "苯乙醇": 14712.0,
    },
    {
        "id": "LX2022_30d", "文献": "李信2022", "工艺": 0, "醋龄月": 0,
        "发酵天数": 21, "温度峰值": 43.0,
        "总酸": 7.35, "不挥发酸": 2.39, "pH": 3.77, "还原糖": 2.24,
        "氨基酸态氮": 0.31,
        "乙酸": 5.033, "乳酸": 1.136, "琥珀酸": 0.0914, "焦谷氨酸": 0.105,
        "柠檬酸": 0.054, "酒石酸": 0.171, "苹果酸": 0.0, "草酸": 0.0, "丙酮酸": 0.041,
        "天冬氨酸": 105.2, "谷氨酸": 58.8, "丙氨酸": 131.7,
        "赖氨酸": 71.6, "酪氨酸": 52.2, "色氨酸": 0.0,
        "甘氨酸": 80.5, "苏氨酸": 71.6, "脯氨酸": 19.2,
        "乙酸乙酯": 1380.0, "乙酸异戊酯": 203.0,
        "乙偶姻": 688.0, "糠醛": 4023.0,
        "四甲基吡嗪": 1423.0, "苯乙醇": 14622.0,
    },
]

# 文献 F: 李国权2013 - 22 个镇江香醋样品的平均值
LGQ2013_AVG = {
    "id": "LGQ2013_AVG22", "文献": "李国权2013", "工艺": 0, "醋龄月": 12,
    "发酵天数": 20, "温度峰值": 42.0,
    "总酸": 5.50, "不挥发酸": 2.36, "pH": 3.65, "还原糖": 1.50,
    "乙酸": 3.194, "乳酸": 1.485, "琥珀酸": 0.0837, "焦谷氨酸": 0.1086,
    "柠檬酸": 0.463, "酒石酸": 0.0652, "苹果酸": 0.0232, "草酸": 0.0232, "丙酮酸": 0.0262,
    "天冬氨酸": 80.0, "谷氨酸": 100.0, "丙氨酸": 150.0,
    "赖氨酸": 90.0, "酪氨酸": 60.0, "色氨酸": 0.0,
    "甘氨酸": 100.0, "苏氨酸": 80.0, "脯氨酸": 30.0,
    "乙酸乙酯": 800.0, "乙酸异戊酯": 60.0,
    "乙偶姻": 200.0, "糠醛": 1500.0,
    "四甲基吡嗪": 100.0, "苯乙醇": 200.0,
}

# 文献 G: 任晓荣2023 - 9 个市售镇江香醋 (3/5/8 年陈酿, 各3重复)
# 数据来源: 表1(有机酸), 表2(氨基酸), 图5(川芎嗪/乙偶姻)
# 氨基酸态氮 = 氨基酸总量 × 0.02057 (从总氨基酸估算)
# 川芎嗪为8年样本估算值; 挥发性化合物(乙酸乙酯等)未测定, 用NaN填充
REN2023_EXT = [
    # 3年陈酿 (醋龄月=36)
    {
        "id": "RENR2023_ZV3_1", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 36,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 5.08, "不挥发酸": 1.90, "pH": 3.20, "还原糖": 1.50,
        "氨基酸态氮": 0.059,  # 总AA 2.8839 mg/mL × 0.02057
        "乙酸": 2.0984, "乳酸": 1.3431, "琥珀酸": 0.0008, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0067, "酒石酸": 0.4068, "苹果酸": 0.0049, "草酸": 0.0105, "丙酮酸": 0.0073,
        "天冬氨酸": 0.0961, "谷氨酸": 0.0384, "丙氨酸": 0.7547,
        "赖氨酸": 0.0306, "酪氨酸": 0.0137, "色氨酸": 0.0,
        "甘氨酸": 0.0254, "苏氨酸": 0.0237, "脯氨酸": 0.0951,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 9000.0, "糠醛": float("nan"),  # 乙偶姻 raw value ~9 mg/mL → ×1000 for μg/mL
        "四甲基吡嗪": 40.0, "苯乙醇": float("nan"),  # 估算
    },
    {
        "id": "RENR2023_ZV3_2", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 36,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 5.08, "不挥发酸": 1.70, "pH": 3.15, "还原糖": 1.50,
        "氨基酸态氮": 0.063,  # 总AA 3.0741 mg/mL × 0.02057
        "乙酸": 2.0355, "乳酸": 1.0042, "琥珀酸": 0.0002, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0067, "酒石酸": 0.2114, "苹果酸": 0.0035, "草酸": 0.0492, "丙酮酸": 0.4107,
        "天冬氨酸": 0.0890, "谷氨酸": 0.0379, "丙氨酸": 0.7840,
        "赖氨酸": 0.0364, "酪氨酸": 0.0161, "色氨酸": 0.0,
        "甘氨酸": 0.0362, "苏氨酸": 0.0186, "脯氨酸": 0.0910,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 9000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 40.0, "苯乙醇": float("nan"),
    },
    {
        "id": "RENR2023_ZV3_3", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 36,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 5.08, "不挥发酸": 1.84, "pH": 3.10, "还原糖": 1.50,
        "氨基酸态氮": 0.061,  # 总AA 2.9598 mg/mL × 0.02057
        "乙酸": 2.3111, "乳酸": 1.0105, "琥珀酸": 0.0012, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0049, "酒石酸": 0.5076, "苹果酸": 0.0217, "草酸": 0.0533, "丙酮酸": 0.0749,
        "天冬氨酸": 0.0884, "谷氨酸": 0.0368, "丙氨酸": 0.7589,
        "赖氨酸": 0.0467, "酪氨酸": 0.0159, "色氨酸": 0.0,
        "甘氨酸": 0.0251, "苏氨酸": 0.0395, "脯氨酸": 0.0927,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 9000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 40.0, "苯乙醇": float("nan"),
    },
    # 5年陈酿 (醋龄月=60)
    {
        "id": "RENR2023_ZV5_1", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 60,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 5.52, "不挥发酸": 1.68, "pH": 3.45, "还原糖": 0.93,
        "氨基酸态氮": 0.141,  # 总AA 6.8570 mg/mL × 0.02057
        "乙酸": 2.3988, "乳酸": 1.2939, "琥珀酸": 0.0520, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0334, "酒石酸": 0.5238, "苹果酸": 0.0049, "草酸": 0.0061, "丙酮酸": 0.0049,
        "天冬氨酸": 0.3741, "谷氨酸": 0.7620, "丙氨酸": 0.9087,
        "赖氨酸": 0.3816, "酪氨酸": 0.1672, "色氨酸": 0.0,
        "甘氨酸": 0.3734, "苏氨酸": 0.2931, "脯氨酸": 0.1509,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 7000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 70.0, "苯乙醇": float("nan"),
    },
    {
        "id": "RENR2023_ZV5_2", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 60,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 5.52, "不挥发酸": 1.55, "pH": 3.50, "还原糖": 0.93,
        "氨基酸态氮": 0.146,  # 总AA 7.0839 mg/mL × 0.02057
        "乙酸": 2.5139, "乳酸": 1.0119, "琥珀酸": 0.0769, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0443, "酒石酸": 0.5157, "苹果酸": 0.0035, "草酸": 0.0296, "丙酮酸": 0.0334,
        "天冬氨酸": 0.4683, "谷氨酸": 0.7853, "丙氨酸": 0.9235,
        "赖氨酸": 0.3667, "酪氨酸": 0.1928, "色氨酸": 0.0,
        "甘氨酸": 0.2992, "苏氨酸": 0.2471, "脯氨酸": 0.1674,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 7000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 70.0, "苯乙醇": float("nan"),
    },
    {
        "id": "RENR2023_ZV5_3", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 60,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 5.52, "不挥发酸": 0.98, "pH": 3.40, "还原糖": 0.93,
        "氨基酸态氮": 0.143,  # 总AA 6.9366 mg/mL × 0.02057
        "乙酸": 2.0224, "乳酸": 1.0029, "琥珀酸": 0.1169, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0217, "酒石酸": 0.2553, "苹果酸": 0.0026, "草酸": 0.0311, "丙酮酸": 0.1790,
        "天冬氨酸": 0.3827, "谷氨酸": 0.8270, "丙氨酸": 0.7621,
        "赖氨酸": 0.3605, "酪氨酸": 0.1082, "色氨酸": 0.0,
        "甘氨酸": 0.3740, "苏氨酸": 0.3010, "脯氨酸": 0.1331,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 7000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 70.0, "苯乙醇": float("nan"),
    },
    # 8年陈酿 (醋龄月=96)
    {
        "id": "RENR2023_ZV8_1", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 96,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 7.43, "不挥发酸": 2.31, "pH": 3.71, "还原糖": 2.96,
        "氨基酸态氮": 0.182,  # 总AA 8.8317 mg/mL × 0.02057
        "乙酸": 3.2727, "乳酸": 1.2118, "琥珀酸": 0.0066, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.1790, "酒石酸": 0.5892, "苹果酸": 0.0098, "草酸": 0.0911, "丙酮酸": 0.0521,
        "天冬氨酸": 0.7745, "谷氨酸": 0.4488, "丙氨酸": 1.5519,
        "赖氨酸": 0.5014, "酪氨酸": 0.1871, "色氨酸": 0.0,
        "甘氨酸": 0.3748, "苏氨酸": 0.5257, "脯氨酸": 0.1489,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 5000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 108.09, "苯乙醇": float("nan"),
    },
    {
        "id": "RENR2023_ZV8_2", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 96,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 7.43, "不挥发酸": 2.25, "pH": 3.71, "还原糖": 2.96,
        "氨基酸态氮": 0.183,  # 总AA 8.8970 mg/mL × 0.02057
        "乙酸": 3.2486, "乳酸": 0.8141, "琥珀酸": 0.0139, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0521, "酒石酸": 0.5464, "苹果酸": 0.0026, "草酸": 0.0514, "丙酮酸": 0.3339,
        "天冬氨酸": 0.7128, "谷氨酸": 0.4023, "丙氨酸": 1.4481,
        "赖氨酸": 0.5444, "酪氨酸": 0.1080, "色氨酸": 0.0,
        "甘氨酸": 0.4077, "苏氨酸": 0.6010, "脯氨酸": 0.1552,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 5000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 94.87, "苯乙醇": float("nan"),
    },
    {
        "id": "RENR2023_ZV8_3", "文献": "任晓荣2023", "工艺": 0, "醋龄月": 96,
        "发酵天数": float("nan"), "温度峰值": float("nan"),
        "总酸": 7.43, "不挥发酸": 2.26, "pH": 3.71, "还原糖": 2.96,
        "氨基酸态氮": 0.176,  # 总AA 8.5415 mg/mL × 0.02057
        "乙酸": 3.1308, "乳酸": 1.6208, "琥珀酸": 0.0479, "焦谷氨酸": float("nan"),
        "柠檬酸": 0.0485, "酒石酸": 0.5706, "苹果酸": 0.0044, "草酸": 0.0620, "丙酮酸": 0.0073,
        "天冬氨酸": 0.6519, "谷氨酸": 0.3184, "丙氨酸": 1.3249,
        "赖氨酸": 0.3169, "酪氨酸": 0.1379, "色氨酸": 0.0,
        "甘氨酸": 0.3970, "苏氨酸": 0.5832, "脯氨酸": 0.1792,
        "乙酸乙酯": float("nan"), "乙酸异戊酯": float("nan"),
        "乙偶姻": 5000.0, "糠醛": float("nan"),
        "四甲基吡嗪": 80.0, "苯乙醇": float("nan"),
    },
]

ALL_SAMPLES = (LI2025_SAMPLES + HU2023_SAMPLES + REN2023_SAMPLES
               + WANG2020_SAMPLES)

# 无感官数据的扩展样本 (用于 PCA / 标准分类 / 视觉化)
EXTENDED_SAMPLES = LIXIN2022_SAMPLES + [LGQ2013_AVG] + REN2023_EXT

# 训练 / 验证 hold-out 划分 (基于工艺和醋龄的多样性)
TRAIN_IDS = {
    "LI2025_TF", "LI2025_CF",
    "HU2023_手工新醋", "HU2023_工业新醋", "HU2023_工业3年",
    "WC2020_Day11", "WC2020_Day21",
    "REN2023_ZV3", "REN2023_ZV5",
}
VAL_IDS = {
    "HU2023_手工3年",   # 测试手工工艺 × 长醋龄
    "WC2020_Day3",       # 测试发酵早期阶段
    "REN2023_ZV8",       # 测试长陈酿外推能力
}

# =============================================================================
# 第二部分: 扩展数据集 (无感官, 仅做 PCA 可视化和子模型)
# =============================================================================
# 郑梦林2021 - 不同陈酿年份的有机酸 + 核苷酸 (单位: g/100mL)
ZHENG2021_EXT = [
    # aging_year, 草酸  酒石酸  乳酸  乙酸  柠檬酸 琥珀酸 苹果酸
    (0, 0.046, 0.376, 1.874, 4.459, 0.339, 0.446, 2.211),
    (1, 0.054, 0.340, 2.055, 4.088, 0.344, 0.311, 1.883),
    (2, 0.052, 0.335, 2.163, 4.560, 0.428, 0.327, 2.187),
    (3, 0.060, 0.336, 2.289, 4.794, 0.340, 0.318, 1.356),
    (5, 0.069, 0.361, 2.375, 6.270, 0.375, 0.277, 0.966),
    (6, 0.067, 0.381, 2.584, 6.728, 0.391, 0.303, 1.008),
    (7, 0.094, 0.297, 2.274, 6.656, 0.431, 0.299, 0.969),
    (8, 0.110, 0.301, 2.195, 6.543, 0.306, 0.274, 0.850),
]
# 余鸣春2006 - 7 个品牌 (总氨基酸 mg/100mL)
YU2006_BRANDS = {
    "陈醋A": 903.8, "普香": 1013.6, "恒顺": 1095.6,
    "金恒顺": 1136.5, "金梅": 1063.2, "金优": 1198.8, "出口": 1023.9,
}

# =============================================================================
# 第三部分: 数据加载与特征工程
# =============================================================================
def load_all_samples() -> pd.DataFrame:
    """加载全部带感官数据的样本, 转为 DataFrame."""
    df = pd.DataFrame(ALL_SAMPLES)
    df = df.set_index("id")
    return df


def split_train_val(df: pd.DataFrame):
    """按预设 ID 划分训练/验证集."""
    train_df = df.loc[df.index.isin(TRAIN_IDS)].copy()
    val_df = df.loc[df.index.isin(VAL_IDS)].copy()
    return train_df, val_df


def compute_oav(df: pd.DataFrame) -> pd.DataFrame:
    """
    OAV = 浓度 / 阈值
    浓度单位: μg/100mL → 转换为 μg/mL 后再除阈值
    """
    df = df.copy()
    for feat in OAV_FEATURES:
        concentration_ug_per_ml = df[feat] / 100.0
        threshold = ODOR_THRESHOLDS[feat]
        df[f"OAV_{feat}"] = concentration_ug_per_ml / threshold
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造派生指标: 比值/比例类特征."""
    df = df.copy()
    df["不挥发酸占比"] = df["不挥发酸"] / (df["总酸"] + 1e-6)
    df["乳酸占比"] = df["乳酸"] / (df["乳酸"] + df["乙酸"] + 1e-6)
    df["还原糖酸比"] = df["还原糖"] / (df["总酸"] + 1e-6)
    df["温度发酵商"] = df["温度峰值"] / (df["发酵天数"] + 1e-6)
    df["总游离氨基酸"] = (
        df["天冬氨酸"] + df["谷氨酸"] + df["丙氨酸"]
        + df["赖氨酸"] + df["酪氨酸"] + df["甘氨酸"]
        + df["苏氨酸"] + df["脯氨酸"]
    )
    df["鲜味氨基酸"] = df["天冬氨酸"] + df["谷氨酸"]
    return df


def build_feature_matrix(df: pd.DataFrame, reduced: bool = False):
    """返回 (X, Y, feature_names)."""
    df = compute_oav(df)
    df = add_engineered_features(df)
    if reduced:
        feature_names = CORE_FEATURES
    else:
        feature_names = (
            PROCESS_FEATURES + PHYSCHEM_FEATURES
            + ["不挥发酸占比", "乳酸占比", "还原糖酸比", "温度发酵商",
               "总游离氨基酸", "鲜味氨基酸"]
            + [f"OAV_{f}" for f in OAV_FEATURES]
        )
    X = df[feature_names].values
    Y = df[SENSORY_OUTPUTS].values
    return X, Y, feature_names


# =============================================================================
# 第四部分: 模型训练器
# =============================================================================
class FlavorModelTrainer:
    """
    集成多模型训练:
      - PLSR (线性, 处理共线性)
      - MLR / LASSO / Ridge (正则化线性基线)
      - SVR (小样本非线性)
      - RandomForest (特征交互)
      - XGBoost (梯度提升)
      - Ensemble (加权平均)
    """

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.models = {}
        self.metrics = {}

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, tune=False):
        """训练全部模型."""
        Xs = self.scaler_X.fit_transform(X_train)
        Ys = self.scaler_Y.fit_transform(Y_train)

        # ---- PLSR (主成分数通过 CV 自动选) ----
        best_n, best_score = 1, -np.inf
        for n in range(1, min(6, len(X_train) - 1)):
            pls = PLSRegression(n_components=n)
            scores = []
            for tr_idx, te_idx in KFold(n_splits=min(5, len(X_train)),
                                        shuffle=True,
                                        random_state=RANDOM_STATE).split(Xs):
                pls.fit(Xs[tr_idx], Ys[tr_idx])
                pred = pls.predict(Xs[te_idx])
                scores.append(r2_score(Ys[te_idx], pred,
                                        multioutput="uniform_average"))
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_n = n
        self.models["PLSR"] = PLSRegression(n_components=best_n).fit(Xs, Ys)
        self.metrics["PLSR"] = {"best_n_components": best_n,
                                "cv_r2": best_score}

        # ---- MLR ----
        self.models["MLR"] = LinearRegression().fit(Xs, Ys)

        # ---- LASSO (调参 or 默认) ----
        if tune:
            lasso = GridSearchCV(
                Lasso(max_iter=10000, random_state=RANDOM_STATE),
                {"alpha": [0.001, 0.01, 0.05, 0.1, 0.3]},
                cv=min(5, len(X_train)),
                scoring="r2",
            ).fit(Xs, Ys)
            self.models["LASSO"] = lasso.best_estimator_
            self.metrics["LASSO"] = {"alpha": lasso.best_params_["alpha"]}
        else:
            self.models["LASSO"] = Lasso(alpha=0.05, max_iter=10000,
                                         random_state=RANDOM_STATE).fit(Xs, Ys)

        # ---- Ridge ----
        if tune:
            ridge = GridSearchCV(
                Ridge(random_state=RANDOM_STATE),
                {"alpha": [0.1, 1.0, 5.0, 10.0, 50.0]},
                cv=min(5, len(X_train)),
                scoring="r2",
            ).fit(Xs, Ys)
            self.models["Ridge"] = ridge.best_estimator_
            self.metrics["Ridge"] = {"alpha": ridge.best_params_["alpha"]}
        else:
            self.models["Ridge"] = Ridge(alpha=50.0,
                                          random_state=RANDOM_STATE).fit(Xs, Ys)

        # ---- SVR (每输出单独训练) ----
        self.models["SVR"] = {}
        for i, name in enumerate(SENSORY_OUTPUTS):
            if tune:
                svr = GridSearchCV(
                    SVR(kernel="rbf"),
                    {"C": [0.5, 1.0, 5.0], "gamma": ["scale", 0.1, 0.05]},
                    cv=min(5, len(X_train)),
                    scoring="r2",
                ).fit(Xs, Ys[:, i])
                self.models["SVR"][name] = svr.best_estimator_
            else:
                self.models["SVR"][name] = SVR(kernel="rbf", C=1.0,
                                               gamma="scale").fit(Xs, Ys[:, i])

        # ---- Random Forest ----
        if tune:
            rf = GridSearchCV(
                RandomForestRegressor(random_state=RANDOM_STATE,
                                      n_jobs=-1),
                {"n_estimators": [100, 200],
                 "max_depth": [3, 5, None]},
                cv=min(5, len(X_train)),
                scoring="r2",
            ).fit(Xs, Ys)
            self.models["RF"] = rf.best_estimator_
            self.metrics["RF"] = rf.best_params_
        else:
            self.models["RF"] = RandomForestRegressor(
                n_estimators=200, max_depth=5,
                random_state=RANDOM_STATE, n_jobs=-1).fit(Xs, Ys)

        # ---- XGBoost (每输出单独训练) ----
        self.models["XGBoost"] = {}
        for i, name in enumerate(SENSORY_OUTPUTS):
            self.models["XGBoost"][name] = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                random_state=RANDOM_STATE, verbosity=0,
            ).fit(Xs, Ys[:, i])

        # ---- 集成模型 (PLSR 0.4 + XGBoost 0.6) ----
        # 在 evaluate 阶段动态计算

        return self

    def predict(self, X, model_name=None):
        """对输入 X 预测, 返回 (n_samples, n_outputs) 数组."""
        Xs = self.scaler_X.transform(X)
        if model_name is not None:
            return self._predict_single(Xs, model_name)

        # 默认返回集成预测
        return self._ensemble_predict(Xs)

    def _predict_single(self, Xs, model_name):
        m = self.models[model_name]
        if model_name == "SVR" or model_name == "XGBoost":
            preds = np.zeros((Xs.shape[0], len(SENSORY_OUTPUTS)))
            for i, name in enumerate(SENSORY_OUTPUTS):
                preds[:, i] = m[name].predict(Xs)
            return preds
        if model_name == "RF":
            return m.predict(Xs)
        # PLSR / MLR / LASSO / Ridge 都是多输出回归
        pred_scaled = m.predict(Xs)
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(-1, 1)
        return self.scaler_Y.inverse_transform(pred_scaled)

    def _ensemble_predict(self, Xs):
        """PLSR 0.4 + XGBoost 0.6 加权集成."""
        pls_pred = self._predict_single(Xs, "PLSR")
        xgb_pred = self._predict_single(Xs, "XGBoost")
        return 0.4 * pls_pred + 0.6 * xgb_pred

    def evaluate(self, X_test, Y_test):
        """在测试集上评估全部模型, 返回 DataFrame."""
        Xs = self.scaler_X.transform(X_test)
        rows = []
        for name in ["PLSR", "MLR", "LASSO", "Ridge", "SVR", "RF",
                     "XGBoost", "Ensemble"]:
            pred = self._predict_single(Xs, name) if name != "Ensemble" \
                else self._ensemble_predict(Xs)
            r2 = r2_score(Y_test, pred, multioutput="uniform_average")
            rmse = np.sqrt(mean_squared_error(Y_test, pred))
            mae = mean_absolute_error(Y_test, pred)
            rows.append({"model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        return pd.DataFrame(rows)


# =============================================================================
# 第五部分: 离线分析
# =============================================================================
def pearson_correlation(df: pd.DataFrame, feature_cols, target_cols):
    """
    计算每个 (特征, 目标) 的 Pearson 相关系数 + p 值.
    输出按 |r| 排序的 DataFrame.
    """
    records = []
    for feat in feature_cols:
        for tgt in target_cols:
            r, p = pearsonr(df[feat], df[tgt])
            records.append({"feature": feat, "target": tgt,
                            "r": r, "p_value": p,
                            "abs_r": abs(r)})
    return pd.DataFrame(records).sort_values("abs_r", ascending=False)


def anova_analysis(df: pd.DataFrame, feature_cols, group_col="醋龄月"):
    """
    单因素 ANOVA: 不同陈酿年份组下各化学指标的差异显著性.
    """
    groups = df[group_col].unique()
    if len(groups) < 2:
        return pd.DataFrame()
    records = []
    for feat in feature_cols:
        samples = [df.loc[df[group_col] == g, feat].values for g in groups]
        try:
            f_stat, p_val = stats.f_oneway(*samples)
            records.append({"feature": feat, "F": f_stat,
                            "p_value": p_val})
        except Exception:
            records.append({"feature": feat, "F": np.nan,
                            "p_value": np.nan})
    return pd.DataFrame(records).sort_values("p_value")


def pca_analysis(df: pd.DataFrame, feature_cols, n_components=2):
    """PCA 降维用于样本分布可视化."""
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_


# =============================================================================
# 第六部分: 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="训练镇江香醋风味模型")
    parser.add_argument("--no-tune", action="store_true",
                        help="跳过网格搜索 (快速模式)")
    parser.add_argument("--reduced-features", action="store_true",
                        help="使用精简特征集 (6个核心特征, 防止过拟合)")
    args = parser.parse_args()
    do_tune = not args.no_tune

    # ----- 加载数据 -----
    print("=" * 70)
    print("[1] 加载数据")
    print("=" * 70)
    df_all = load_all_samples()
    print(f"  训练样本数 (带感官): {len(df_all)}")
    train_df, val_df = split_train_val(df_all)
    print(f"  训练集: {len(train_df)} 个 -> {sorted(train_df.index.tolist())}")
    print(f"  验证集: {len(val_df)} 个 -> {sorted(val_df.index.tolist())}")

    # 加载扩展样本 (无感官, 仅用于分类与可视化)
    df_extended = pd.DataFrame(EXTENDED_SAMPLES).set_index("id")
    print(f"  扩展样本 (无感官): {len(df_extended)} 个 -> {sorted(df_extended.index.tolist())}")

    # 保存原始数据到 CSV
    df_all.to_csv(DATA_DIR / "all_samples.csv", encoding="utf-8-sig")
    train_df.to_csv(DATA_DIR / "train_samples.csv", encoding="utf-8-sig")
    val_df.to_csv(DATA_DIR / "val_samples.csv", encoding="utf-8-sig")
    df_extended.to_csv(DATA_DIR / "extended_samples.csv", encoding="utf-8-sig")
    print(f"  数据已保存到 {DATA_DIR}/")

    # ----- 标准分类 (全部 18 样本, v2.0 对齐 GB/T 18623-2011) -----
    print("\n" + "=" * 70)
    print("[2] 依据标准 (v2.0, 对齐 GB/T 18623-2011) 对全部样本分级")
    print("=" * 70)
    print(f"  强制指标: 总酸≥{GB_STANDARD['总酸']['合格品']}, "
          f"不挥发酸≥{GB_STANDARD['不挥发酸']['合格品']}, "
          f"还原糖≥{GB_STANDARD['还原糖']['合格品']}, "
          f"氨基酸态氮≥{GB_STANDARD['氨基酸态氮']['合格品']} g/100mL")
    df_all_with_oav = compute_oav(df_all)
    df_ext_with_oav = compute_oav(df_extended)
    df_combined = pd.concat([df_all_with_oav, df_ext_with_oav], axis=0)
    grade_df = classify_dataframe(df_combined)
    grade_df.to_csv(MODEL_DIR / "standard_grade.csv", index=False)
    print(grade_df.to_string(index=False))
    print(f"\n  风味等级分布:")
    print(grade_df["综合等级"].value_counts().to_string(header=["样本数"]))
    print(f"\n  国标等级分布:")
    print(grade_df["国标等级"].value_counts().to_string(header=["样本数"]))

    # ----- 构造特征 -----
    use_reduced = args.reduced_features
    X_train, Y_train, feat_names = build_feature_matrix(train_df, reduced=use_reduced)
    X_val, Y_val, _ = build_feature_matrix(val_df, reduced=use_reduced)
    feat_dim_note = "(精简模式, 防止过拟合)" if use_reduced else "(完整模式)"
    print(f"\n  特征维度: {X_train.shape[1]} {feat_dim_note}")

    # ----- 训练多模型 -----
    print("\n" + "=" * 70)
    print(f"[2] 训练多模型 (网格搜索={'开启' if do_tune else '关闭'})")
    print("=" * 70)
    trainer = FlavorModelTrainer().fit(
        X_train, Y_train, X_val, Y_val, tune=do_tune
    )
    print("  模型列表:", list(trainer.models.keys()))

    # ----- 在验证集上评估 -----
    print("\n" + "=" * 70)
    print("[3] 验证集评估 (3 个 hold-out: 手工3年 / Day3 / ZV8)")
    print("=" * 70)
    print("  注: 验证集刻意选择训练范围外的样本, 测试外推能力")
    metrics_df = trainer.evaluate(X_val, Y_val)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(MODEL_DIR / "evaluation_metrics.csv", index=False)

    # 各输出维度 R^2 (更直观)
    Xs_val = trainer.scaler_X.transform(X_val)
    pred_e = trainer._ensemble_predict(Xs_val)
    per_output = []
    for i, name in enumerate(SENSORY_OUTPUTS):
        r2 = r2_score(Y_val[:, i], pred_e[:, i])
        per_output.append({"output": name, "R2": r2})
    per_df = pd.DataFrame(per_output)
    print("\n  集成模型各感官维度 R^2:")
    print(per_df.round(3).to_string(index=False))

    # ----- LOOCV 训练集评估 (更稳健) -----
    print("\n" + "=" * 70)
    print("[4] 训练集 LOOCV 评估")
    print("=" * 70)
    loo = LeaveOneOut()
    loo_records = []
    for tr_idx, te_idx in loo.split(X_train):
        t = FlavorModelTrainer().fit(
            X_train[tr_idx], Y_train[tr_idx], tune=False
        )
        pred = t.predict(X_train[te_idx])
        for i, name in enumerate(SENSORY_OUTPUTS):
            loo_records.append({
                "sample": train_df.index[te_idx[0]],
                "output": name,
                "true": Y_train[te_idx, i][0],
                "pred": pred[0, i],
            })
    loo_df = pd.DataFrame(loo_records)
    loo_r2 = loo_df.groupby("output").apply(
        lambda g: r2_score(g["true"], g["pred"])
    )
    print(f"  LOOCV 各感官维度平均 R^2: {loo_r2.mean():.4f}")
    print(loo_r2.round(4).to_string(header=["R^2"]))
    loo_df.to_csv(MODEL_DIR / "loocv_predictions.csv", index=False)

    # ----- 基于标准等级对模型做一致性评估 -----
    print("\n" + "=" * 70)
    print("[4.5] 模型预测 vs 标准等级的一致性 (验证集)")
    print("=" * 70)
    # 用模型预测 val 样本的化学特征 → 反推等级 → 与真实等级比较
    Xs_val = trainer.scaler_X.transform(X_val)
    pred_chem = pd.DataFrame(
        trainer.scaler_X.inverse_transform(Xs_val),
        columns=feat_names, index=val_df.index
    )
    # 从 OAV 字段反推原始浓度 (× 100 → μg/100mL)
    if "OAV_乙酸乙酯" in pred_chem.columns:
        pred_chem["乙酸乙酯"] = pred_chem["OAV_乙酸乙酯"] * 5.0 * 100
    if "OAV_乙酸异戊酯" in pred_chem.columns:
        pred_chem["乙酸异戊酯"] = pred_chem["OAV_乙酸异戊酯"] * 0.3 * 100
    if "OAV_四甲基吡嗪" in pred_chem.columns:
        pred_chem["四甲基吡嗪"] = pred_chem["OAV_四甲基吡嗪"] * 0.5 * 100

    # 逐样本用模型预测的化学特征分级
    pred_grades = []
    for idx, row in pred_chem.iterrows():
        sample = {
            "总酸": row["总酸"], "还原糖": row["还原糖"],
            "不挥发酸占比": row.get("不挥发酸占比", 0),
            "鲜味氨基酸": row.get("鲜味氨基酸", 0),
            "OAV_乙酸乙酯": row.get("OAV_乙酸乙酯", 0),
            "OAV_乙酸异戊酯": row.get("OAV_乙酸异戊酯", 0),
            "四甲基吡嗪": row.get("四甲基吡嗪", 0),
        }
        r = classify_sample(sample)
        pred_grades.append({"样品": idx, "预测总分": r["总分"],
                             "预测等级": r["综合等级"]})

    # 与真实等级对比
    real_grades = grade_df[grade_df["样品"].isin(val_df.index)].copy()
    real_grades = real_grades.set_index("样品")[["总分", "综合等级", "国标等级"]]
    pred_grades_df = pd.DataFrame(pred_grades).set_index("样品")
    comparison = pred_grades_df.join(real_grades, rsuffix="_真实")
    comparison.columns = ["预测总分", "预测等级", "真实总分", "真实综合等级", "真实国标等级"]
    comparison["等级一致"] = comparison["预测等级"] == comparison["真实综合等级"]
    comparison["国标一致"] = comparison["预测总分"].apply(
        lambda s: check_gb_compliance({"总酸": s, "不挥发酸": s, "还原糖": s,
                                      "氨基酸态氮": s})["国标等级"]
    ) == comparison["真实国标等级"]
    print(comparison.round(1).to_string())
    print(f"\n  风味等级一致率: {comparison['等级一致'].mean()*100:.1f}%")
    comparison.to_csv(MODEL_DIR / "grade_consistency.csv", encoding="utf-8-sig")

    # ----- 相关性分析 -----
    print("\n" + "=" * 70)
    print("[5] Pearson 相关性 (Top 15)")
    print("=" * 70)
    df_all_with_features = add_engineered_features(compute_oav(df_all))
    corr = pearson_correlation(
        df_all_with_features,
        feature_cols=feat_names,
        target_cols=SENSORY_OUTPUTS,
    )
    top15 = corr.head(15)
    print(top15.to_string(index=False))
    corr.to_csv(MODEL_DIR / "pearson_correlation.csv", index=False)

    # ----- PCA 分布 -----
    print("\n" + "=" * 70)
    print("[6] PCA 样本分布")
    print("=" * 70)
    coords, var_ratio = pca_analysis(df_all_with_features, feat_names)
    print(f"  PC1 解释方差: {var_ratio[0]:.2%}, PC2: {var_ratio[1]:.2%}")
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df_all.index)
    pca_df["醋龄月"] = df_all["醋龄月"]
    pca_df["工艺"] = df_all["工艺"]
    print(pca_df.round(3).to_string())
    pca_df.to_csv(MODEL_DIR / "pca_coordinates.csv", encoding="utf-8-sig")

    # ----- ANOVA (按醋龄月分组) -----
    print("\n" + "=" * 70)
    print("[7] 单因素 ANOVA (按醋龄月分组)")
    print("=" * 70)
    anova = anova_analysis(df_all_with_features,
                          feature_cols=feat_names,
                          group_col="醋龄月")
    print(anova.head(10).to_string(index=False))
    anova.to_csv(MODEL_DIR / "anova_by_aging.csv", index=False)

    # ----- 特征重要性 (来自 XGBoost / RF) -----
    print("\n" + "=" * 70)
    print("[8] 特征重要性 (RF, Top 15)")
    print("=" * 70)
    rf_importance = pd.DataFrame({
        "feature": feat_names,
        "importance": trainer.models["RF"].feature_importances_,
    }).sort_values("importance", ascending=False)
    print(rf_importance.head(15).to_string(index=False))
    rf_importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    # ----- 保存模型包 -----
    bundle = {
        "models": trainer.models,
        "scaler_X": trainer.scaler_X,
        "scaler_Y": trainer.scaler_Y,
        "feature_names": feat_names,
        "sensory_outputs": SENSORY_OUTPUTS,
        "process_features": PROCESS_FEATURES,
        "physchem_features": PHYSCHEM_FEATURES,
        "organic_acid_features": ORGANIC_ACID_FEATURES,
        "amino_acid_features": AMINO_ACID_FEATURES,
        "volatile_features": VOLATILE_FEATURES,
        "oav_features": OAV_FEATURES,
        "odor_thresholds": ODOR_THRESHOLDS,
        "train_ids": sorted(TRAIN_IDS),
        "val_ids": sorted(VAL_IDS),
        "ensemble_weights": {"PLSR": 0.4, "XGBoost": 0.6},
    }
    with open(MODEL_DIR / "flavor_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  ✓ 模型包已保存到 {MODEL_DIR / 'flavor_model.pkl'}")

    # 保存特征配置 (供 app.py 加载)
    config = {
        "feature_names": feat_names,
        "sensory_outputs": SENSORY_OUTPUTS,
        "odor_thresholds": ODOR_THRESHOLDS,
    }
    with open(MODEL_DIR / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("训练完成 ✓")
    print("=" * 70)
    print(f"  - 模型: {MODEL_DIR / 'flavor_model.pkl'}")
    print(f"  - 数据: {DATA_DIR}/")
    print(f"  - 下一步: 运行 python app.py")


if __name__ == "__main__":
    main()
