#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standard.py - 镇江香醋风味评价标准 v2.0

按 风味评价标准.md v2.0 实现
对齐 GB/T 18623-2011《地理标志产品 镇江香醋》
"""

import json
from pathlib import Path

import pandas as pd


# =============================================================================
# 国标 GB/T 18623-2011 强制指标阈值
# =============================================================================
GB_STANDARD = {
    "总酸":      {"单位": "g/100mL", "合格品": 3.5,  "优级品": 4.5},
    "不挥发酸":  {"单位": "g/100mL", "合格品": 0.5,  "优级品": 0.5},
    "还原糖":    {"单位": "g/100mL", "合格品": 1.0,  "优级品": 1.0},
    "氨基酸态氮": {"单位": "g/100mL", "合格品": 0.18, "优级品": 0.18},
}


# =============================================================================
# 综合等级阈值
# =============================================================================
GRADE_THRESHOLDS = [
    (90, "★★★★★ 特级陈酿"),
    (75, "★★★★ 优级"),
    (60, "★★★ 合格"),
    (45, "★★ 普通"),
    (0,  "★ 次品"),
]


# =============================================================================
# 各维度评分函数 (按 v2.0 标准)
# =============================================================================
def score_acidity(total_acid):
    """酸度分 (max 25)."""
    if total_acid >= 7.0: return 25, "★★★★"
    if total_acid >= 6.0: return 22, "★★★"
    if total_acid >= 5.0: return 20, "★★★"
    if total_acid >= 4.5: return 17, "★★"  # 国标优级
    if total_acid >= 3.5: return 10, "★"   # 国标合格
    return 0, "✗"


def score_nonvolatile_acid(nonvolatile):
    """不挥发酸分 (max 15). 单位 g/100mL"""
    if nonvolatile >= 2.0: return 15, "★★★"
    if nonvolatile >= 1.0: return 12, "★★"
    if nonvolatile >= 0.5: return 8,  "★"  # 国标达标
    return 0, "✗"


def score_reducing_sugar(rs):
    """还原糖分 (max 10)."""
    if rs >= 2.5: return 10, "★★★"
    if rs >= 2.0: return 8,  "★★"
    if rs >= 1.5: return 6,  "★★"
    if rs >= 1.0: return 4,  "★"  # 国标达标
    return 0, "✗"


def score_amino_nitrogen(an):
    """氨基酸态氮分 (max 15). 单位 g/100mL"""
    if an >= 0.30: return 15, "★★★"
    if an >= 0.20: return 12, "★★"
    if an >= 0.18: return 8,  "★"  # 国标达标
    return 0, "✗"


def score_mildness(nonvolatile_ratio):
    """柔和度分 (max 10). 不挥发酸占比"""
    pct = nonvolatile_ratio * 100
    if pct >= 30: return 10, "★★★"
    if pct >= 25: return 7,  "★★"
    if pct >= 20: return 4,  "★"
    return 0, "✗"


def score_ester(oav_total):
    """酯香分 (max 15)."""
    if oav_total >= 15: return 15, "★★★"
    if oav_total >= 5:  return 10, "★★"
    if oav_total >= 1:  return 5,  "★"
    return 0, "✗"


def score_ttmp(ttp_conc):
    """川芎嗪分 (max 10)."""
    if ttp_conc >= 80: return 10, "★★★"
    if ttp_conc >= 40: return 7,  "★★"
    if ttp_conc >= 20: return 4,  "★★"
    return 0, "★"


SCORERS = {
    "酸度":       (score_acidity, 25),
    "不挥发酸":   (score_nonvolatile_acid, 15),
    "还原糖":     (score_reducing_sugar, 10),
    "氨基酸态氮": (score_amino_nitrogen, 15),
    "柔和度":     (score_mildness, 10),
    "酯香":       (score_ester, 15),
    "川芎嗪":     (score_ttmp, 10),
}

# 顺序: 必须与权重对应
DIMENSION_ORDER = ["酸度", "不挥发酸", "还原糖", "氨基酸态氮",
                   "柔和度", "酯香", "川芎嗪"]


# =============================================================================
# 国标符合性检验
# =============================================================================
def check_gb_compliance(sample):
    """
    检验样本是否满足 GB/T 18623-2011 的 4 项强制指标.

    sample 必需字段: 总酸, 不挥发酸, 还原糖, 氨基酸态氮
    """
    out = {}
    for metric, threshold in GB_STANDARD.items():
        v = float(sample.get(metric, 0))
        合格 = v >= threshold["合格品"]
        优级 = v >= threshold["优级品"]
        if not 合格:
            level = "不合格"
        elif 优级:
            level = "优级"
        else:
            level = "合格"
        out[metric] = {
            "实测值": round(v, 3),
            "合格阈值": threshold["合格品"],
            "优级阈值": threshold["优级品"],
            "达合格": 合格,
            "达优级": 优级,
            "等级": level,
        }

    n_合格 = sum(1 for m in out.values() if m["达合格"])
    n_优级 = sum(1 for m in out.values() if m["达优级"])

    if n_合格 == 4 and n_优级 == 4:
        gb_grade = "优级"
    elif n_合格 == 4:
        gb_grade = "合格"
    else:
        gb_grade = "不合格"

    return {
        **out,
        "国标等级": gb_grade,
        "4项_全合格": n_合格 == 4,
        "4项_全优级": n_优级 == 4,
        "合格项数": n_合格,
        "优级项数": n_优级,
    }


# =============================================================================
# 样本预处理
# =============================================================================
def prepare_sample(df_row):
    """
    把原始样本转成标准输入.
    自动计算派生字段:
      - 氨基酸态氮: 若未直接给出, 从已有 9 种氨基酸估算
        估算公式: 氨基酸态氮 = 总氨基酸(mg/100mL) × 0.078 / 1000 (g/100mL)
        (基于王超2020实测: 0.29 g/100mL 对应 ~3.7 g/100mL 总氨基酸, 系数≈0.078)
      - 鲜味氨基酸 = 谷氨酸 + 天冬氨酸
      - 不挥发酸占比 = 不挥发酸 / 总酸
      - OAV_乙酸乙酯 = (乙酸乙酯 / 100) / 5.0
      - OAV_乙酸异戊酯 = (乙酸异戊酯 / 100) / 0.3
    """
    s = dict(df_row)
    tot = float(s.get("总酸", 1))
    if tot > 0:
        s["不挥发酸占比"] = float(s.get("不挥发酸", 0)) / tot
    else:
        s["不挥发酸占比"] = 0
    s["鲜味氨基酸"] = float(s.get("谷氨酸", 0)) + float(s.get("天冬氨酸", 0))
    # 氨基酸态氮估算 (优先用实测值, 否则从 9 种氨基酸求和估算)
    if "氨基酸态氮" not in s:
        AA_FIELDS = ["天冬氨酸", "谷氨酸", "丙氨酸", "赖氨酸",
                     "酪氨酸", "色氨酸", "甘氨酸", "苏氨酸", "脯氨酸"]
        total_aa = sum(float(s.get(f, 0)) for f in AA_FIELDS)  # mg/100mL
        s["氨基酸态氮"] = total_aa * 0.078 / 1000.0  # g/100mL
    s["OAV_乙酸乙酯"] = float(s.get("乙酸乙酯", 0)) / 100.0 / 5.0
    s["OAV_乙酸异戊酯"] = float(s.get("乙酸异戊酯", 0)) / 100.0 / 0.3
    return s


# =============================================================================
# 主分类函数
# =============================================================================
def classify_sample(sample):
    """
    完整分级 + 国标符合性.
    """
    s = prepare_sample(sample)
    detail = {}
    total = 0
    for dim in DIMENSION_ORDER:
        scorer, cap = SCORERS[dim]
        if dim == "酸度":
            v = s.get("总酸", 0)
        elif dim == "不挥发酸":
            v = s.get("不挥发酸", 0)
        elif dim == "还原糖":
            v = s.get("还原糖", 0)
        elif dim == "氨基酸态氮":
            v = s.get("氨基酸态氮", 0)
        elif dim == "柔和度":
            v = s.get("不挥发酸占比", 0)
        elif dim == "酯香":
            v = s.get("OAV_乙酸乙酯", 0) + s.get("OAV_乙酸异戊酯", 0)
        elif dim == "川芎嗪":
            v = s.get("四甲基吡嗪", 0)
        else:
            continue
        s_val, g = scorer(v)
        detail[dim] = {"分值": s_val, "上限": cap, "等级": g, "原始值": round(v, 3)}
        total += s_val

    grade = _grade(total)
    gb = check_gb_compliance(s)

    return {
        "总分": total,
        "满分": sum(cap for _, cap in SCORERS.values()),
        "综合等级": grade,
        "国标等级": gb["国标等级"],
        "国标详情": gb,
        "明细": detail,
    }


def classify_dataframe(df):
    """对 DataFrame 逐行分类, 返回分级 + 国标符合性表."""
    records = []
    for idx, row in df.iterrows():
        r = classify_sample(row.to_dict())
        rec = {
            "样品": idx,
            "总分": r["总分"],
            "综合等级": r["综合等级"],
            "国标等级": r["国标等级"],
        }
        for dim, info in r["明细"].items():
            rec[f"{dim}_分"] = info["分值"]
            rec[f"{dim}_等级"] = info["等级"]
        rec["4项_合格"] = r["国标详情"]["合格项数"]
        rec["4项_优级"] = r["国标详情"]["优级项数"]
        records.append(rec)
    return pd.DataFrame(records)


def _grade(total):
    for thr, label in GRADE_THRESHOLDS:
        if total >= thr:
            return label
    return GRADE_THRESHOLDS[-1][1]


# =============================================================================
# 演示
# =============================================================================
if __name__ == "__main__":
    test = {
        "总酸": 7.20, "不挥发酸": 2.21, "还原糖": 2.64,
        "谷氨酸": 300, "天冬氨酸": 27, "氨基酸态氮": 0.21,
        "乙酸乙酯": 2699, "乙酸异戊酯": 80, "四甲基吡嗪": 30,
    }
    r = classify_sample(test)
    print(f"总分: {r['总分']}/{r['满分']}  等级: {r['综合等级']}  国标: {r['国标等级']}")
    print("\n明细:")
    for k, v in r["明细"].items():
        print(f"  {k:<8} {v['分值']:>2}/{v['上限']:<2}  {v['等级']}  ({v['原始值']})")
    print("\n国标详情:")
    for k, v in r["国标详情"].items():
        if isinstance(v, dict):
            print(f"  {k}: {v['等级']}  ({v['实测值']})")
        else:
            print(f"  {k}: {v}")
