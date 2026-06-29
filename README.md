# 镇江香醋风味评分系统

Zhenjiang Aromatic Vinegar (ZAV) Flavor Scoring System

## 项目概述

基于文献数据（4篇核心论文）构建的镇江香醋风味评分系统，采用**规则引擎+线性校准**混合架构，在小样本(n=9)条件下实现稳定可解释的风味预测。

## 核心指标

| 指标 | 值 |
|------|-----|
| 校准 R² | 0.67 |
| 校准参数 α | 0.9906 |
| 校准参数 β | +1.2503 |
| 感官维度 | 11维 |

## 快速开始

### Web界面
```bash
pip install flask numpy pandas scikit-learn
python web_app.py
# 浏览器打开 http://127.0.0.1:5000
```

### 命令行
```bash
# 规则模式 (推荐)
python app.py --mode rule --input sample.json

# 机器学习模式
python app.py --mode ml --input sample.json
```

## 文件结构

```
├── web_app.py       # Web服务入口
├── rule_engine.py   # 规则引擎核心
├── standard.py      # GB/T 18623-2011 标准分类
├── app.py           # CLI应用入口
├── tran.py          # 模型训练脚本
└── data/            # 数据目录
```

## 算法说明

### 规则引擎
基于以下文献构建专家规则：
- Zhang Ning (2025): ZAV感官特征 / Table 5氨基酸
- Li Guoping (2026): Y1/Y3/Y5/Y10 OAV变化趋势
- GB/T 18623-2011: 国标分级阈值

### 评分维度 (11维)
酸味、苦味、甜味、咸味、风味、酱香、谷物香、炒米香、米醋香、持久度、柔和度

## 模型说明

本项目在小样本(n=9)条件下进行了深入分析：

- **LOOCV R² = -5.83**: 证明机器学习模型在小样本下完全无效
- **规则引擎 R² = 0.67**: 仅用2个参数(α,β)达到稳定预测
- **PCA验证通过**: 高年份样本落在"高风味物质"象限

## License

MIT
