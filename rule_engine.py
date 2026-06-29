#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rule_engine.py - 镇江香醋风味评分规则引擎

基于文献共识构建的专家规则系统:
  - Zhang Ning (2025): ZAV感官特征, Table 5氨基酸
  - Li Guoping (2026): Y1/Y3/Y5/Y10 OAV变化趋势 (Fig3)
  - GB/T 18623-2011: 国标分级阈值

规则引擎 + 线性校准: 物理可解释 + 数据适配
  alpha = 0.9906, beta = 1.2503 (n=9 OLS拟合)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


class ZAVScoringSystem:
    """镇江香醋风味评分系统 v2.0 - 规则引擎+线性校准"""

    REASONABLE_RANGES = {
        "醋龄月": (0, 120),
        "总酸": (3.0, 10.0),
        "不挥发酸": (0.5, 3.5),
        "还原糖": (0.5, 5.0),
        "总游离氨基酸": (0.1, 10.0),
        "乙酸乙酯": (100, 5000),
        "四甲基吡嗪": (5, 200),
        "乙酸": (0.5, 8.0),
        "pH": (2.0, 5.5),
    }

    FIELD_NAMES = {
        "醋龄月": "醋龄",
        "总酸": "总酸",
        "不挥发酸": "不挥发酸",
        "还原糖": "还原糖",
        "总游离氨基酸": "总游离氨基酸",
        "乙酸乙酯": "乙酸乙酯",
        "四甲基吡嗪": "四甲基吡嗪",
        "乙酸": "乙酸",
        "pH": "pH值",
    }

    def __init__(self, alpha=0.9906, beta=1.2503):
        self.alpha = alpha
        self.beta = beta
        self.fitted = True

    def validate_sample(self, sample: dict) -> list:
        """校验输入样本, 返回警告列表 (字段名, 当前值, 范围)"""
        warnings = []
        for key, (lo, hi) in self.REASONABLE_RANGES.items():
            if key not in sample:
                continue
            val = sample[key]
            if pd.isna(val):
                continue
            if val < lo or val > hi:
                name = self.FIELD_NAMES.get(key, key)
                warnings.append(
                    f"{name}={val:.2f}超出合理范围[{lo}, {hi}], 评分可能不可靠"
                )
        return warnings

    def compute_base(self, sample: dict, use_ph: bool = True) -> dict:
        """计算规则基础分 (物理/化学可解释)"""
        months = sample.get("醋龄月", 0)
        ethyl_val = sample.get("乙酸乙酯", np.nan)
        tmp = sample.get("四甲基吡嗪", np.nan)
        total_acid = sample.get("总酸", np.nan)
        reducing_sugar = sample.get("还原糖", np.nan)
        total_aa = sample.get("总游离氨基酸", np.nan)
        ph = sample.get("pH", np.nan)

        # f_age: 陈酿贡献 (边际递减饱和)
        if months <= 0:
            fa = 2.0
        elif months <= 12:
            fa = 2.0 + (months / 12) * 2.0
        elif months <= 36:
            fa = 4.0 + ((months - 12) / 24) * 2.0
        elif months <= 60:
            fa = 6.0 + ((months - 36) / 24) * 2.5
        elif months <= 96:
            fa = 8.5 + ((months - 60) / 36) * 1.0
        else:
            fa = 9.5

        # f_ethyl: 乙酸乙酯OAV (钟形曲线, Y5峰值)
        if pd.isna(ethyl_val):
            fe = 5.0
        else:
            oav = ethyl_val / 5.0
            if oav < 1.0:
                fe = oav * 4.0
            elif oav <= 30:
                fe = 4.0 + ((oav - 1) / 29) * 4.0
            elif oav <= 60:
                fe = 8.0 - ((oav - 30) / 30) * 2.0
            elif oav <= 100:
                fe = 6.0 - ((oav - 60) / 40) * 3.0
            else:
                fe = max(1.0, 3.0 - (oav - 100) * 0.02)

        # f_tmp: 四甲基吡嗪 (单调递增)
        if pd.isna(tmp):
            ft = 5.0
        elif tmp < 10:
            ft = tmp / 10.0 * 3.0
        elif tmp < 30:
            ft = 3.0 + ((tmp - 10) / 20) * 2.0
        elif tmp < 60:
            ft = 5.0 + ((tmp - 30) / 30) * 2.5
        elif tmp < 100:
            ft = 7.5 + ((tmp - 60) / 40) * 2.0
        else:
            ft = min(10.0, 9.5 + (tmp - 100) * 0.005)

        # f_acidity: 总酸
        if pd.isna(total_acid):
            fac = 5.0
        elif total_acid < 4.5:
            fac = max(1.0, total_acid / 4.5 * 4.0)
        elif total_acid < 6.0:
            fac = 4.0 + (total_acid - 4.5) / 1.5 * 2.0
        elif total_acid <= 8.0:
            fac = 6.0 + (total_acid - 6.0) / 2.0 * 3.0
        else:
            fac = max(6.0, 9.0 - (total_acid - 8.0) * 0.5)

        # f_sugar: 还原糖
        if pd.isna(reducing_sugar):
            fs = 5.0
        elif reducing_sugar < 1.0:
            fs = max(1.0, reducing_sugar * 4.0)
        elif reducing_sugar < 2.0:
            fs = 4.0 + (reducing_sugar - 1.0) * 2.0
        elif reducing_sugar <= 4.0:
            fs = 6.0 + (reducing_sugar - 2.0) / 2.0 * 3.0
        else:
            fs = min(10.0, 9.0 + (reducing_sugar - 4.0) * 0.1)

        # f_umami: 鲜味氨基酸
        if pd.isna(total_aa):
            fu = 5.0
        elif total_aa < 0.3:
            fu = max(1.0, total_aa / 0.3 * 3.0)
        elif total_aa < 1.0:
            fu = 3.0 + (total_aa - 0.3) / 0.7 * 2.0
        elif total_aa <= 3.0:
            fu = 5.0 + (total_aa - 1.0) / 2.0 * 4.0
        else:
            fu = min(10.0, 9.0 + (total_aa - 3.0) * 0.2)

        # f_process: 工艺加成 (固态发酵=1 > 封闭式=0)
        # 训练数据: 固态工艺样本的酱香/谷物香/风味普遍高0.5-1.0分
        process = sample.get("工艺", 1)
        if pd.isna(process) or process == 1:
            fproc = 7.0
        else:
            fproc = 5.5

        # f_ph: pH舒适度 (影响柔和感/刺激感)
        # 镇江香醋理想范围 pH 3.0-3.8, 低于3.0过酸, 高于3.8过淡
        if pd.isna(ph) or not use_ph:
            fph = 0.0
            ph_active = False
            ph_warning = None
        else:
            ph_active = True
            if ph < 2.0 or ph > 5.5:
                fph = 0.0
                ph_warning = f"pH={ph:.2f}超出合理范围[2.0, 5.5], pH维度计0分"
            elif ph < 2.5:
                fph = 2.0
                ph_warning = None
            elif ph < 3.0:
                fph = 4.0 + (ph - 2.5) * 2.0
                ph_warning = None
            elif ph <= 3.5:
                fph = 9.0 + (ph - 3.0) * 2.0
                ph_warning = None
            elif ph <= 3.8:
                fph = 10.0
                ph_warning = None
            elif ph <= 4.2:
                fph = 10.0 - (ph - 3.8) * 3.75
                ph_warning = None
            elif ph <= 5.0:
                fph = 6.0
                ph_warning = None
            else:
                fph = 3.0
                ph_warning = f"pH={ph:.2f}偏高(>5.0), 评分降低"

        w = {"f_age": 0.25, "f_ethyl": 0.16, "f_tmp": 0.13,
             "f_acidity": 0.13, "f_sugar": 0.08, "f_umami": 0.08,
             "f_proc": 0.10,
             "f_ph": 0.07 if use_ph else 0.0}

        base = (fa * w["f_age"] + fe * w["f_ethyl"] + ft * w["f_tmp"]
                + fac * w["f_acidity"] + fs * w["f_sugar"] + fu * w["f_umami"]
                + fproc * w["f_proc"]
                + (fph * w["f_ph"] if use_ph else 0.0))

        return {
            "base": base,
            "components": {
                "f_age": fa, "f_ethyl": fe, "f_tmp": ft,
                "f_acidity": fac, "f_sugar": fs, "f_umami": fu,
                "f_proc": fproc,
                "f_ph": fph, "f_ph_active": ph_active,
            },
            "weights": w,
            "ph_warning": ph_warning,
        }

    def _raw_to_sensory(self, raw: dict, process: int = 1) -> dict:
        """规则分 → 11维感官空间
        process: 0=封闭式, 1=固态/手工 (固态发酵对酱香/谷物香有加成)
        """
        c = raw["components"]
        fa = c["f_age"]
        fe = c["f_ethyl"]
        ft = c["f_tmp"]
        fac = c["f_acidity"]
        fs = c["f_sugar"]
        fu = c["f_umami"]
        fph = c["f_ph"]
        ph_active = c.get("f_ph_active", False)

        ph_bonus_soft = fph * 0.3 if ph_active else 0.0
        ph_penalty_sour = fph * 0.1 if ph_active else 0.0

        # 工艺加成: 固态发酵(1) > 封闭式(0)
        # 基于训练数据: 固态工艺样本的酱香/谷物香/风味比封闭式高0.5-1.0
        process_bonus = 0.5 if process == 1 else 0.0
        jiang_bonus = 0.4 * process_bonus
        guwu_bonus = 0.3 * process_bonus
        fengwei_bonus = 0.2 * process_bonus

        mapped = {
            "s_醋酸味": np.clip(fac * 0.6 + fa * 0.2 + fu * 0.2 - ph_penalty_sour, 1.0, 10.0),
            "s_苦味": np.clip(8.0 - min(5.0, fa * 0.5 + fac * 0.3), 1.0, 10.0),
            "s_甜味": np.clip(fs * 0.5 + fu * 0.3 + fe * 0.2, 1.0, 10.0),
            "s_咸味": np.clip(4.0 + fac * 0.1, 1.0, 10.0),
            "s_风味": np.clip(fa * 0.3 + fe * 0.25 + ft * 0.25 + fac * 0.1 + fs * 0.1 + fengwei_bonus, 1.0, 10.0),
            "s_酱香": np.clip(ft * 0.7 + fa * 0.2 + fu * 0.1 + jiang_bonus, 1.0, 10.0),
            "s_谷物香": np.clip(fe * 0.5 + fa * 0.3 + ft * 0.2 + guwu_bonus, 1.0, 10.0),
            "s_炒米香": np.clip(ft * 0.5 + fa * 0.3 + fac * 0.2, 1.0, 10.0),
            "s_米醋香": np.clip(fac * 0.6 + fe * 0.3 + fs * 0.1, 1.0, 10.0),
            "s_持久度": np.clip(fa * 0.6 + ft * 0.2 + fac * 0.2, 1.0, 10.0),
            "s_柔和度": np.clip(fe * 0.35 + fa * 0.25 + fs * 0.15 + fac * 0.1 + ph_bonus_soft, 1.0, 10.0),
        }
        return mapped

    def predict(self, sample: dict, explain: bool = True, use_ph: bool = True) -> dict:
        """
        评分预测

        参数:
          sample: dict, 包含醋龄月/总酸/乙酸乙酯/四甲基吡嗪/还原糖/总游离氨基酸/工艺
          explain: bool, 是否输出特征贡献度分解
          use_ph: bool, 是否启用pH维度评分 (默认开启)

        返回:
          dict: 包含11维感官评分 + 综合得分 + 等级 + (可选)特征贡献 + 警告列表
        """
        warnings = self.validate_sample(sample)
        process = int(sample.get("工艺", 1))
        raw = self.compute_base(sample, use_ph=use_ph)
        base = raw["base"]
        calibrated = self.alpha * base + self.beta

        sensory = self._raw_to_sensory(raw, process=process)

        w = raw["weights"]
        c = raw["components"]
        contrib = {
            "陈酿贡献": round(c["f_age"] * w["f_age"], 3),
            "酯香贡献": round(c["f_ethyl"] * w["f_ethyl"], 3),
            "酱香贡献": round(c["f_tmp"] * w["f_tmp"], 3),
            "酸度贡献": round(c["f_acidity"] * w["f_acidity"], 3),
            "甜味贡献": round(c["f_sugar"] * w["f_sugar"], 3),
            "鲜味贡献": round(c["f_umami"] * w["f_umami"], 3),
            "工艺贡献": round(c["f_proc"] * w["f_proc"], 3),
        }
        if use_ph and c.get("f_ph_active", False):
            contrib["pH贡献"] = round(c["f_ph"] * w["f_ph"], 3)

        contrib["校准偏移"] = round(self.beta, 3)
        contrib["规则基础分"] = round(base, 3)
        contrib["校准后分"] = round(calibrated, 3)

        result = {
            **{k: round(v, 2) for k, v in sensory.items()},
            "综合得分": round(calibrated, 2),
            "等级": self._grade(calibrated),
            "_use_ph": use_ph,
            "_process": process,
            "warnings": warnings,
        }
        if raw.get("ph_warning"):
            result["ph_warning"] = raw["ph_warning"]
        if explain:
            result["特征贡献"] = contrib

        return result

    def predict_from_df(self, df: pd.DataFrame, explain: bool = False, use_ph: bool = True) -> pd.DataFrame:
        """DataFrame批量预测"""
        results = []
        for idx, row in df.iterrows():
            r = self.predict(row.to_dict(), explain=explain, use_ph=use_ph)
            r["样品"] = idx
            results.append(r)
        return pd.DataFrame(results).set_index("样品")

    def _grade(self, score):
        if score >= 9.0:
            return "★★★★★ 顶级陈酿"
        if score >= 8.0:
            return "★★★★ 优质陈酿"
        if score >= 7.0:
            return "★★★ 合格品"
        if score >= 6.0:
            return "★★ 普通品"
        return "★ 次品"

    @staticmethod
    def fit_calibration(df: pd.DataFrame) -> dict:
        """
        用n=9样本重新拟合 alpha/beta (未来数据扩充时可重调用)
        仅2自由度，不会过拟合
        """
        sensory_cols = ["s_风味", "s_柔和度", "s_持久度", "s_醋酸味", "s_甜味"]
        scorer = ZAVScoringSystem()

        base_scores = []
        true_scores = []
        for _, row in df.iterrows():
            bs = scorer.compute_base(row.to_dict())["base"]
            true_s = np.mean([row[c] for c in sensory_cols])
            base_scores.append(bs)
            true_scores.append(true_s)

        X = np.array(base_scores).reshape(-1, 1)
        y = np.array(true_scores)
        lr = LinearRegression().fit(X, y)

        ss_res = np.sum((y - lr.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        return {"alpha": lr.coef_[0], "beta": lr.intercept_, "R2": r2}

    @staticmethod
    def pca_validate(df: pd.DataFrame) -> pd.DataFrame:
        """PCA验证: 评分与化学分布一致性"""
        feature_cols = ["总酸", "不挥发酸", "还原糖", "乙酸乙酯",
                        "四甲基吡嗪", "醋龄月"]
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(df[available].median())
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)

        scorer = ZAVScoringSystem()
        result = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df.index)
        result["醋龄月"] = df["醋龄月"].values
        result["基础分"] = [
            scorer.compute_base(row.to_dict())["base"] for _, row in df.iterrows()
        ]
        result["校准分"] = [
            scorer.predict(row.to_dict(), explain=False)["综合得分"]
            for _, row in df.iterrows()
        ]
        return result, pca
