#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py - 镇江香醋风味模型应用脚本

负责:
  1. 加载 tran.py 训练好的模型包
  2. 单样本预测: 输入工艺+理化+风味 → 输出 11 维感官评分
  3. 批量预测: 读取 CSV 文件, 逐行预测
  4. 工艺反演: 给定目标风味 → 推荐最优工艺参数

运行:
  python app.py                       # 运行默认演示
  python app.py --input sample.json    # 从 JSON 读取输入
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
BUNDLE_PATH = MODEL_DIR / "flavor_model.pkl"
CONFIG_PATH = MODEL_DIR / "feature_config.json"


# =============================================================================
# 预测器类
# =============================================================================
class FlavorPredictor:
    """封装模型加载、特征工程、预测与工艺优化."""

    # 综合得分权重
    COMP_WEIGHTS = {
        "醋酸味": 0.25, "柔和度": 0.25, "持久度": 0.20,
        "风味": 0.15, "甜味": 0.15,
    }

    # 等级阈值
    GRADES = [
        (9.0, "★★★★★ 顶级陈酿"),
        (8.0, "★★★★ 优质陈酿"),
        (7.0, "★★★ 合格品"),
        (6.0, "★★ 普通品"),
        (0.0, "★ 次品"),
    ]

    def __init__(self, bundle_path=BUNDLE_PATH):
        with open(bundle_path, "rb") as f:
            self.bundle = pickle.load(f)
        self.models = self.bundle["models"]
        self.scaler_X = self.bundle["scaler_X"]
        self.scaler_Y = self.bundle["scaler_Y"]
        self.feature_names = self.bundle["feature_names"]
        self.sensory_outputs = self.bundle["sensory_outputs"]
        self.oav_features = self.bundle["oav_features"]
        self.odor_thresholds = self.bundle["odor_thresholds"]
        self.weights = self.bundle["ensemble_weights"]

    # ------------------------------------------------------------------
    # 特征工程
    # ------------------------------------------------------------------
    def _build_X(self, input_dict):
        """把单条记录转换为特征向量."""
        d = dict(input_dict)
        # OAV 变换 (浓度 μg/100mL → μg/mL → / 阈值)
        for feat in self.oav_features:
            conc_ug_ml = d[feat] / 100.0
            d[f"OAV_{feat}"] = conc_ug_ml / self.odor_thresholds[feat]
        # 派生特征
        d["不挥发酸占比"] = d["不挥发酸"] / (d["总酸"] + 1e-6)
        d["乳酸占比"] = d["乳酸"] / (d["乳酸"] + d["乙酸"] + 1e-6)
        d["还原糖酸比"] = d["还原糖"] / (d["总酸"] + 1e-6)
        d["温度发酵商"] = d["温度峰值"] / (d["发酵天数"] + 1e-6)
        d["总游离氨基酸"] = (
            d["天冬氨酸"] + d["谷氨酸"] + d["丙氨酸"]
            + d["赖氨酸"] + d["酪氨酸"] + d["甘氨酸"]
            + d["苏氨酸"] + d["脯氨酸"]
        )
        d["鲜味氨基酸"] = d["天冬氨酸"] + d["谷氨酸"]
        return np.array([d[f] for f in self.feature_names]).reshape(1, -1)

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict(self, input_dict, return_full=False):
        """输入字典, 返回 11 维感官评分 + 综合得分."""
        X = self._build_X(input_dict)
        Xs = self.scaler_X.transform(X)

        # 集成预测 (PLSR 0.4 + XGBoost 0.6)
        pls_pred = self.scaler_Y.inverse_transform(
            self.models["PLSR"].predict(Xs)
        )
        xgb_preds = np.zeros((1, len(self.sensory_outputs)))
        for i, name in enumerate(self.sensory_outputs):
            xgb_preds[0, i] = float(self.models["XGBoost"][name].predict(Xs[0:1])[0])
        pred = self.weights["PLSR"] * pls_pred + self.weights["XGBoost"] * xgb_preds
        pred = np.clip(pred, 1.0, 8.0)  # 限制在合理区间

        result = {name.replace("s_", ""): round(float(pred[0, i]), 2)
                  for i, name in enumerate(self.sensory_outputs)}

        # 综合得分: 0-8 分 → 0-10 分
        weighted_sum = sum(result[k] * w for k, w in self.COMP_WEIGHTS.items())
        result["综合风味得分"] = round(weighted_sum / 8.0 * 10.0, 2)
        result["等级"] = self._grade(result["综合风味得分"])

        if return_full:
            result["_raw"] = {"PLSR": pls_pred.tolist()[0],
                              "XGBoost": xgb_preds.tolist()[0]}
        return result

    def predict_batch(self, df):
        """对 DataFrame 每行预测, 返回结果 DataFrame."""
        results = []
        for idx, row in df.iterrows():
            r = self.predict(row.to_dict())
            results.append(r)
        return pd.DataFrame(results)

    def predict_from_csv(self, csv_path):
        """读取 CSV 批量预测."""
        df = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig")
        return self.predict_batch(df)

    # ------------------------------------------------------------------
    # 工艺反演
    # ------------------------------------------------------------------
    def optimize_process(self, target_attr="柔和度", target_value=8.0,
                         current=None, method="nelder"):
        """
        给定目标感官分值, 反演最优工艺+理化+风味参数.
        默认优化 8 个关键变量, 其它参数使用 current 或默认值.
        """
        if current is None:
            current = {
                "工艺": 0, "醋龄月": 12, "发酵天数": 16, "温度峰值": 43.0,
                "总酸": 7.0, "不挥发酸": 2.0, "pH": 3.85, "还原糖": 2.6,
                "乙酸": 5.0, "乳酸": 2.5, "琥珀酸": 0.4, "焦谷氨酸": 0.15,
                "柠檬酸": 0.30, "酒石酸": 0.30, "苹果酸": 0.035, "草酸": 0.060, "丙酮酸": 0.030,
                "天冬氨酸": 20.0, "谷氨酸": 150.0, "丙氨酸": 80.0,
                "赖氨酸": 35.0, "酪氨酸": 28.0, "色氨酸": 100.0,
                "甘氨酸": 25.0, "苏氨酸": 20.0, "脯氨酸": 28.0,
                "乙酸乙酯": 1500.0, "乙酸异戊酯": 60.0,
                "乙偶姻": 1000.0, "糠醛": 2000.0,
                "四甲基吡嗪": 50.0, "苯乙醇": 100.0,
            }

        # 待优化变量索引与边界
        opt_keys = ["醋龄月", "发酵天数", "温度峰值", "总酸", "不挥发酸",
                    "乙酸乙酯", "乙偶姻", "四甲基吡嗪"]
        bounds = [
            (0, 96),       # 醋龄月
            (12, 22),      # 发酵天数
            (38, 46),      # 温度峰值
            (5.5, 8.5),    # 总酸
            (1.5, 3.0),    # 不挥发酸
            (500, 3000),   # 乙酸乙酯
            (200, 2500),   # 乙偶姻
            (10, 100),     # 四甲基吡嗪
        ]

        target_idx = None
        for i, name in enumerate(self.sensory_outputs):
            if target_attr in name:
                target_idx = i
                break
        if target_idx is None:
            raise ValueError(f"未找到感官维度: {target_attr}")

        def neg_objective(x):
            trial = dict(current)
            for k, v in zip(opt_keys, x):
                trial[k] = float(v)
            pred = self.predict(trial)
            return - (pred[target_attr] - target_value) ** 2

        if method == "nelder":
            # Nelder-Mead 不强制约束, 这里做裁剪以防越界
            def clipped_obj(x):
                clipped = [min(max(x[i], bounds[i][0]), bounds[i][1])
                           for i in range(len(x))]
                return neg_objective(clipped)
            x0 = [current[k] for k in opt_keys]
            res = minimize(clipped_obj, x0, method="Nelder-Mead",
                           options={"maxiter": 300, "xatol": 0.5})
        else:
            res = differential_evolution(neg_objective, bounds,
                                          maxiter=200, tol=1e-4,
                                          seed=42)

        optimal = dict(current)
        for k, v in zip(opt_keys, res.x):
            optimal[k] = float(v)
        optimal_pred = self.predict(optimal)

        # 最终裁剪到合理区间
        for k, (lo, hi) in zip(opt_keys, bounds):
            optimal[k] = float(np.clip(optimal[k], lo, hi))
        optimal_pred = self.predict(optimal)

        return {
            "目标": f"{target_attr}={target_value}",
            "推荐工艺": {k: round(optimal[k], 2) for k in opt_keys},
            "预测感官": {k.replace("s_", ""): optimal_pred[k.replace("s_", "")]
                        for k in self.sensory_outputs},
            "综合得分": optimal_pred["综合风味得分"],
            "等级": optimal_pred["等级"],
        }

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _grade(self, score):
        for threshold, label in self.GRADES:
            if score >= threshold:
                return label
        return self.GRADES[-1][1]

    def feature_importance(self):
        """返回 RF 特征重要性 (来自训练时保存的 bundle)."""
        if "RF" in self.models:
            imp = self.models["RF"].feature_importances_
            return sorted(zip(self.feature_names, imp),
                         key=lambda x: -x[1])
        return []


# =============================================================================
# 演示流程
# =============================================================================
def demo_single(predictor):
    """演示单样本预测."""
    print("=" * 70)
    print("[演示 1] 单样本打分 - 预测一款新型封闭式发酵陈醋")
    print("=" * 70)
    sample = {
        "工艺": 0, "醋龄月": 18, "发酵天数": 16, "温度峰值": 43.5,
        "总酸": 7.30, "不挥发酸": 2.15, "pH": 3.82, "还原糖": 2.70,
        "乙酸": 5.40, "乳酸": 3.00, "琥珀酸": 0.55, "焦谷氨酸": 0.17,
        "柠檬酸": 0.38, "酒石酸": 0.30, "苹果酸": 0.038, "草酸": 0.062, "丙酮酸": 0.030,
        "天冬氨酸": 25.0, "谷氨酸": 250.0, "丙氨酸": 85.0,
        "赖氨酸": 32.0, "酪氨酸": 25.0, "色氨酸": 140.0,
        "甘氨酸": 25.0, "苏氨酸": 20.0, "脯氨酸": 28.0,
        "乙酸乙酯": 2200.0, "乙酸异戊酯": 70.0,
        "乙偶姻": 1800.0, "糠醛": 1500.0,
        "四甲基吡嗪": 35.0, "苯乙醇": 100.0,
    }
    result = predictor.predict(sample)
    print("\n  输入: 封闭式发酵 16 天, 陈酿 18 个月")
    print("\n  11 维感官评分:")
    for k, v in result.items():
        if k not in ("综合风味得分", "等级", "_raw"):
            bar = "█" * int(v) + "░" * (8 - int(v))
            print(f"    {k:<8} {v:5.2f}  {bar}")
    print(f"\n  综合风味得分: {result['综合风味得分']} / 10")
    print(f"  等级评定:    {result['等级']}")
    return result


def demo_batch(predictor):
    """演示批量预测."""
    print("\n" + "=" * 70)
    print("[演示 2] 批量预测 - 读取 CSV")
    print("=" * 70)
    csv_in = DATA_DIR / "val_samples.csv"
    if not csv_in.exists():
        print(f"  [跳过] 未找到 {csv_in}")
        return None
    df_in = pd.read_csv(csv_in, index_col=0, encoding="utf-8-sig")
    print(f"  输入: {len(df_in)} 个样本 (来自验证集)")
    preds = predictor.predict_from_csv(csv_in)
    preds.insert(0, "样品", df_in.index.tolist())
    print("\n  预测结果 (部分维度):")
    print(preds[["样品", "醋酸味", "柔和度", "持久度",
                "风味", "综合风味得分", "等级"]].to_string(index=False))
    return preds


def demo_optimization(predictor):
    """演示工艺反演."""
    print("\n" + "=" * 70)
    print("[演示 3] 工艺反演 - 求解'柔和度 ≥ 8.5'的最优工艺")
    print("=" * 70)
    result = predictor.optimize_process(
        target_attr="柔和度", target_value=8.5
    )
    print(f"\n  目标: {result['目标']}")
    print("\n  推荐工艺参数:")
    for k, v in result["推荐工艺"].items():
        print(f"    {k:<14} {v}")
    print(f"\n  预测感官: 柔和度={result['预测感官']['柔和度']}, "
          f"持久度={result['预测感官']['持久度']}")
    print(f"  综合得分: {result['综合得分']} → {result['等级']}")
    return result


def demo_feature_importance(predictor):
    """演示特征重要性."""
    print("\n" + "=" * 70)
    print("[演示 4] 特征重要性 (RF Top 10)")
    print("=" * 70)
    for feat, imp in predictor.feature_importance()[:10]:
        print(f"  {feat:<18} {imp:.4f}")


def main():
    parser = argparse.ArgumentParser(description="镇江香醋风味模型应用")
    parser.add_argument("--mode", default="rule",
                        choices=["rule", "ml"],
                        help="评分模式: rule=规则引擎(默认), ml=机器学习")
    parser.add_argument("--input", type=str, help="从 JSON 读取输入")
    parser.add_argument("--target", default="柔和度",
                        help="工艺优化目标维度 (仅ml模式)")
    parser.add_argument("--value", type=float, default=8.5,
                        help="工艺优化目标值 (仅ml模式)")
    parser.add_argument("--no-demo", action="store_true",
                        help="跳过演示, 仅加载模型")
    args = parser.parse_args()

    print("=" * 70)
    print("镇江香醋风味模型 v2.0")
    print("=" * 70)

    if args.mode == "rule":
        from rule_engine import ZAVScoringSystem
        scorer = ZAVScoringSystem()

        print("  模式: rule (规则引擎+线性校准)")
        print(f"  校准参数: alpha={scorer.alpha:.4f}, beta={scorer.beta:.4f}")

        if args.input:
            with open(args.input, encoding="utf-8") as f:
                sample = json.load(f)
            result = scorer.predict(sample, explain=True)
            print("\n输入样本预测结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        df_all = pd.read_csv(DATA_DIR / "train_samples.csv", encoding="utf-8-sig", index_col=0)
        print(f"\n训练集(n={len(df_all)}) 规则评分演示:")
        print("-" * 70)
        for _, row in df_all.iterrows():
            r = scorer.predict(row.to_dict(), explain=False)
            true_s = np.mean([row[c] for c in ["s_风味", "s_柔和度", "s_持久度", "s_醋酸味", "s_甜味"]])
            diff = r["综合得分"] - true_s
            print(f"  {row.name:20s} 醋龄={row['醋龄月']:3.0f}月 | "
                  f"规则:{r['综合得分']:.2f} 真实:{true_s:.2f} | 差:{diff:+.2f} | {r['等级']}")

        print("\n" + "=" * 70)
        print("规则引擎使用说明")
        print("=" * 70)
        print("  python app.py --mode rule --input sample.json")
        print("  python app.py --mode rule --input sample.json --explain  # 详细特征贡献")
        return

    predictor = FlavorPredictor()
    print(f"  模式: ml (机器学习)")
    print(f"  模型已加载: {BUNDLE_PATH.name}")
    print(f"  特征数: {len(predictor.feature_names)}")
    print(f"  感官维度: {len(predictor.sensory_outputs)}")

    if args.no_demo:
        return

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            sample = json.load(f)
        result = predictor.predict(sample)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    demo_single(predictor)
    demo_batch(predictor)
    demo_optimization(predictor)
    demo_feature_importance(predictor)

    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print("  1. 规则模式(推荐): python app.py --mode rule --input your_sample.json")
    print("  2. ML模式:        python app.py --input your_sample.json")
    print("  3. 输入字段: 工艺/醋龄月/总酸/乙酸/谷氨酸/乙酸乙酯/...")
    print("  4. 完整字段列表见 feature_config.json")


if __name__ == "__main__":
    main()
