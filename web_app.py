#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_app.py - 镇江香醋风味评分系统 Web界面

运行:
    python web_app.py
    打开浏览器访问 http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pandas as pd
from rule_engine import ZAVScoringSystem

app = Flask(__name__)
scorer = ZAVScoringSystem()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>镇江香醋风味评分系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        h1 {
            font-size: 2.2em;
            color: #e8d5b7;
            margin-bottom: 10px;
            letter-spacing: 4px;
        }
        .subtitle {
            color: #8b8b8b;
            font-size: 0.95em;
        }
        .badge {
            display: inline-block;
            background: rgba(232, 213, 183, 0.15);
            border: 1px solid #e8d5b7;
            border-radius: 20px;
            padding: 4px 16px;
            font-size: 0.8em;
            color: #e8d5b7;
            margin-top: 10px;
        }

        .main-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        @media (max-width: 800px) {
            .main-layout { grid-template-columns: 1fr; }
        }

        .panel {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 28px;
            backdrop-filter: blur(10px);
        }
        .panel-title {
            font-size: 1.1em;
            color: #e8d5b7;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(232,213,183,0.2);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .panel-title::before {
            content: '';
            width: 4px;
            height: 18px;
            background: #c9a96e;
            border-radius: 2px;
        }

        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            color: #a0a0a0;
            font-size: 0.85em;
            margin-bottom: 6px;
        }
        input[type="number"], select {
            width: 100%;
            padding: 10px 14px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 0.95em;
            transition: border-color 0.3s;
        }
        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #c9a96e;
        }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }

        .toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 10px 16px;
            margin-bottom: 16px;
        }
        .toggle-label {
            font-size: 0.88em;
            color: #a0a0a0;
        }
        .toggle-label span {
            color: #c9a96e;
            font-size: 0.78em;
            margin-left: 6px;
        }
        .toggle-switch {
            position: relative;
            width: 44px;
            height: 24px;
        }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            inset: 0;
            background: rgba(255,255,255,0.15);
            border-radius: 24px;
            transition: 0.3s;
        }
        .toggle-slider::before {
            content: '';
            position: absolute;
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background: white;
            border-radius: 50%;
            transition: 0.3s;
        }
        .toggle-switch input:checked + .toggle-slider { background: #c9a96e; }
        .toggle-switch input:checked + .toggle-slider::before { transform: translateX(20px); }

        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #c9a96e 0%, #a07d4a 100%);
            border: none;
            border-radius: 10px;
            color: #1a1a2e;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.3s, transform 0.2s;
            margin-top: 10px;
        }
        .btn:hover { opacity: 0.88; transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }

        .result-placeholder {
            text-align: center;
            color: #555;
            padding: 60px 0;
            font-size: 0.95em;
        }

        #result {
            display: none;
        }
        .score-display {
            text-align: center;
            padding: 20px 0 10px;
        }
        .score-number {
            font-size: 4em;
            font-weight: 700;
            color: #e8d5b7;
            line-height: 1;
        }
        .score-grade {
            font-size: 1.3em;
            margin-top: 8px;
        }
        .grade-stars { color: #c9a96e; }
        .score-label {
            color: #666;
            font-size: 0.85em;
            margin-top: 4px;
        }

        .sensory-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 20px;
        }
        .sensory-item {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 12px 14px;
        }
        .sensory-item .name {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 6px;
        }
        .sensory-item .bar-wrap {
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            height: 8px;
        }
        .sensory-item .bar {
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #c9a96e, #e8d5b7);
            transition: width 0.6s ease;
        }
        .sensory-item .value {
            font-size: 0.9em;
            color: #ccc;
            margin-top: 4px;
        }

        .warning-box {
            display: none;
            background: rgba(255, 200, 100, 0.12);
            border: 1px solid rgba(255, 200, 100, 0.4);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 14px;
            font-size: 0.85em;
            color: #f0c060;
        }
        .warning-box.show { display: block; }
        .warning-box ul { margin: 4px 0 0 0; padding-left: 20px; }

        .contribution-section {
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.08);
        }
        .contrib-title {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 14px;
        }
        .contrib-item {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            gap: 10px;
        }
        .contrib-item .name {
            width: 80px;
            font-size: 0.82em;
            color: #888;
        }
        .contrib-item .bar-wrap {
            flex: 1;
            background: rgba(255,255,255,0.08);
            border-radius: 4px;
            height: 10px;
        }
        .contrib-item .bar {
            height: 10px;
            border-radius: 4px;
            background: #c9a96e;
        }
        .contrib-item .value {
            width: 50px;
            text-align: right;
            font-size: 0.82em;
            color: #ccc;
        }

        .info-row {
            display: flex;
            gap: 16px;
            margin-top: 20px;
        }
        .info-chip {
            flex: 1;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        .info-chip .val {
            font-size: 1.1em;
            color: #e8d5b7;
        }
        .info-chip .lbl {
            font-size: 0.75em;
            color: #666;
            margin-top: 4px;
        }

        .quick-presets {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        .preset-btn {
            padding: 5px 14px;
            background: rgba(201,169,110,0.2);
            border: 1px solid rgba(201,169,110,0.4);
            border-radius: 20px;
            color: #c9a96e;
            font-size: 0.8em;
            cursor: pointer;
            transition: background 0.2s;
        }
        .preset-btn:hover { background: rgba(201,169,110,0.35); }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>镇江香醋风味评分系统</h1>
        <div class="subtitle">Zhenjiang Aromatic Vinegar Flavor Scoring System</div>
        <div class="badge">规则引擎 + 线性校准 | α=0.9906 β=1.2503</div>
    </header>

    <div class="main-layout">
        <!-- 左侧: 输入表单 -->
        <div class="panel">
            <div class="panel-title">样品参数输入</div>

            <div class="quick-presets">
                <span style="color:#666;font-size:0.82em;align-self:center;margin-right:4px;">快速预设:</span>
                <button class="preset-btn" onclick="preset('新醋')">新醋 (0月)</button>
                <button class="preset-btn" onclick="preset('3年陈')">3年陈 (36月)</button>
                <button class="preset-btn" onclick="preset('5年陈')">5年陈 (60月)</button>
                <button class="preset-btn" onclick="preset('8年陈')">8年陈 (96月)</button>
            </div>

            <form id="scoreForm">
                <div class="row">
                    <div class="form-group">
                        <label>醋龄月 <span style="color:#888;font-size:0.85em">(0-120)</span></label>
                        <input type="number" name="醋龄月" id="醋龄月" value="60" min="0" max="120">
                    </div>
                    <div class="form-group">
                        <label>工艺</label>
                        <select name="工艺" id="工艺">
                            <option value="1">固态发酵</option>
                            <option value="0">封闭式</option>
                        </select>
                    </div>
                </div>

                <div class="row">
                    <div class="form-group">
                        <label>总酸 (g/100mL) <span style="color:#888;font-size:0.85em">(3-10)</span></label>
                        <input type="number" name="总酸" id="总酸" value="6.32" step="0.01" min="3" max="10">
                    </div>
                    <div class="form-group">
                        <label>不挥发酸 (g/100mL) <span style="color:#888;font-size:0.85em">(0.5-3.5)</span></label>
                        <input type="number" name="不挥发酸" id="不挥发酸" value="1.85" step="0.01" min="0.5" max="3.5">
                    </div>
                </div>

                <div class="row">
                    <div class="form-group">
                        <label>还原糖 (g/100mL) <span style="color:#888;font-size:0.85em">(0.5-5)</span></label>
                        <input type="number" name="还原糖" id="还原糖" value="0.93" step="0.01" min="0.5" max="5">
                    </div>
                    <div class="form-group">
                        <label>总游离氨基酸 (g/100mL) <span style="color:#888;font-size:0.85em">(0.1-10)</span></label>
                        <input type="number" name="总游离氨基酸" id="总游离氨基酸" value="4.0" step="0.01" min="0.1" max="10">
                    </div>
                </div>

                <div class="row">
                    <div class="form-group">
                        <label>乙酸乙酯 (μg/mL) <span style="color:#888;font-size:0.85em">(100-5000)</span></label>
                        <input type="number" name="乙酸乙酯" id="乙酸乙酯" value="1500" step="1" min="100" max="5000">
                    </div>
                    <div class="form-group">
                        <label>四甲基吡嗪 (μg/mL) <span style="color:#888;font-size:0.85em">(5-200)</span></label>
                        <input type="number" name="四甲基吡嗪" id="四甲基吡嗪" value="44" step="1" min="5" max="200">
                    </div>
                </div>

                <div class="row">
                    <div class="form-group">
                        <label>乙酸 (g/100mL) <span style="color:#888;font-size:0.85em">(0.5-8)</span></label>
                        <input type="number" name="乙酸" id="乙酸" value="2.31" step="0.01" min="0.5" max="8">
                    </div>
                    <div class="form-group">
                        <label>pH值 <span style="color:#888;font-size:0.85em">(2.0-5.5)</span></label>
                        <input type="number" name="pH" id="pH" value="3.65" step="0.01" min="2.0" max="5.5">
                    </div>
                </div>

                <div class="toggle-row">
                    <div class="toggle-label">
                        pH维度评分
                        <span>(开启后影响柔和度/刺激感)</span>
                    </div>
                    <label class="toggle-switch">
                        <input type="checkbox" id="use_ph" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>

                <button type="submit" class="btn">🏆 开始评分</button>
            </form>
        </div>

        <!-- 右侧: 结果展示 -->
        <div class="panel">
            <div class="panel-title">评分结果</div>

            <div id="placeholder" class="result-placeholder">
                <div style="font-size:3em;margin-bottom:16px;">🍶</div>
                请填写左侧参数并点击"开始评分"<br>
                <small style="color:#555;margin-top:8px;display:block;">或点击快速预设快速体验</small>
            </div>

            <div id="result">
                <div class="score-display">
                    <div class="score-number" id="综合得分">--</div>
                    <div class="score-grade"><span class="grade-stars" id="等级">--</span></div>
                    <div class="score-label">综合风味得分</div>
                </div>

                <div class="info-row">
                    <div class="info-chip">
                        <div class="val" id="规则基础分">--</div>
                        <div class="lbl">规则基础分</div>
                    </div>
                    <div class="info-chip">
                        <div class="val" id="校准偏移">--</div>
                        <div class="lbl">校准偏移</div>
                    </div>
                    <div class="info-chip">
                        <div class="val" id="陈酿月">--</div>
                        <div class="lbl">陈酿月数</div>
                    </div>
                </div>

                <div class="warning-box" id="warningBox">
                    ⚠ <span id="warningTitle">输入数据警告:</span>
                    <ul id="warningList"></ul>
                </div>

                <div class="sensory-grid" id="sensoryGrid"></div>

                <div class="contribution-section">
                    <div class="contrib-title">特征贡献度分解</div>
                    <div id="contribList"></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
const PRESETS = {
    '新醋': {醋龄月:0,总酸:7.2,不挥发酸:2.3,pH:3.85,还原糖:2.6,总游离氨基酸:3.0,乙酸乙酯:600,四甲基吡嗪:20,乙酸:5.5,工艺:1},
    '3年陈': {醋龄月:36,总酸:5.8,不挥发酸:1.8,pH:3.9,还原糖:1.8,总游离氨基酸:3.5,乙酸乙酯:1100,四甲基吡嗪:42,乙酸:3.1,工艺:1},
    '5年陈': {醋龄月:60,总酸:6.32,不挥发酸:1.85,pH:3.65,还原糖:0.93,总游离氨基酸:4.0,乙酸乙酯:1500,四甲基吡嗪:44,乙酸:2.31,工艺:1},
    '8年陈': {醋龄月:96,总酸:7.43,不挥发酸:2.3,pH:3.71,还原糖:2.96,总游离氨基酸:5.5,乙酸乙酯:1800,四甲基吡嗪:95,乙酸:3.22,工艺:1},
};

function preset(name) {
    const p = PRESETS[name];
    for (const [k, v] of Object.entries(p)) {
        const el = document.getElementById(k);
        if (el) el.value = v;
    }
}

document.getElementById('scoreForm').onsubmit = async function(e) {
    e.preventDefault();
    const fd = new FormData(this);
    const data = {};
    for (const [k, v] of fd.entries()) {
        data[k] = parseFloat(v) || 0;
    }
    data.use_ph = document.getElementById('use_ph').checked;

    const resp = await fetch('/api/score', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    const r = await resp.json();

    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('result').style.display = 'block';

    document.getElementById('综合得分').textContent = r.综合得分.toFixed(1);
    document.getElementById('等级').textContent = r.等级;
    document.getElementById('规则基础分').textContent = r.特征贡献.规则基础分.toFixed(2);
    document.getElementById('校准偏移').textContent = '+' + r.特征贡献.校准偏移.toFixed(2);
    document.getElementById('陈酿月').textContent = data.醋龄月 + '月';

    const warningBox = document.getElementById('warningBox');
    const warningList = document.getElementById('warningList');
    warningList.innerHTML = '';
    const warnings = r.warnings || [];
    if (r.ph_warning) warnings.push(r.ph_warning);
    if (warnings.length > 0) {
        warnings.forEach(w => {
            const li = document.createElement('li');
            li.textContent = w;
            warningList.appendChild(li);
        });
        warningBox.classList.add('show');
    } else {
        warningBox.classList.remove('show');
    }

    const sensoryMap = {
        's_醋酸味':'酸味','s_苦味':'苦味','s_甜味':'甜味','s_咸味':'咸味',
        's_风味':'风味','s_酱香':'酱香','s_谷物香':'谷物香','s_炒米香':'炒米香',
        's_米醋香':'米醋香','s_持久度':'持久度','s_柔和度':'柔和度'
    };
    const grid = document.getElementById('sensoryGrid');
    grid.innerHTML = '';
    for (const [k, name] of Object.entries(sensoryMap)) {
        const v = r[k] || 0;
        grid.innerHTML += `
            <div class="sensory-item">
                <div class="name">${name}</div>
                <div class="bar-wrap"><div class="bar" style="width:${v*10}%"></div></div>
                <div class="value">${v.toFixed(1)}</div>
            </div>`;
    }

    const contribMap = {
        '陈酿贡献':'陈酿','酯香贡献':'酯香(OAV)','酱香贡献':'酱香(TMP)',
        '酸度贡献':'酸度','甜味贡献':'甜味','鲜味贡献':'鲜味',
        '工艺贡献':'工艺加成','pH贡献':'pH舒适度'
    };
    const contribList = document.getElementById('contribList');
    contribList.innerHTML = '';
    const maxContrib = 3.0;
    for (const [k, name] of Object.entries(contribMap)) {
        const v = r.特征贡献[k] || 0;
        const pct = Math.min(100, (v / maxContrib) * 100);
        contribList.innerHTML += `
            <div class="contrib-item">
                <div class="name">${name}</div>
                <div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div>
                <div class="value">+${v.toFixed(2)}</div>
            </div>`;
    }
};
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/score', methods=['POST'])
def api_score():
    data = request.get_json()
    use_ph = data.pop('use_ph', True)
    result = scorer.predict(data, explain=True, use_ph=use_ph)
    return jsonify(result)


if __name__ == '__main__':
    print("=" * 60)
    print("镇江香醋风味评分系统 Web界面")
    print("启动中: http://127.0.0.1:5000")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
