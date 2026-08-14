"""MulitiModal Demo — Gradio 前端界面
========================================

布局结构:
  ┌──────────────────────────────────────────────────┐
  │ MulitiModal 标题栏 + 管线示意图                    │
  ├──────────┬──────────┬───────────────────────────┤
  │ 场景选择  │ 输入面板  │ 推理结果面板               │
  │          │ 波形图    │ ┌─────┐ ┌─────┐          │
  │          │ 参数表    │ │TCM  │ │动态  │          │
  │          │ 诊断特征  │ │概率条│ │表征  │          │
  │          │          │ └─────┘ └─────┘          │
  │          │          │ ┌─────┐ ┌─────────────┐  │
  │          │          │ │EEG  │ │ 推荐方案     │  │
  │          │          │ │表征  │ │ + 力度       │  │
  │          │          │ └─────┘ └─────────────┘  │
  └──────────┴──────────┴───────────────────────────┘
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────
# 确保路径正确
# ──────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(DEMO_DIR))

import gradio as gr  # noqa: E402
from app import get_manager, CONSTITUTION_NAMES  # noqa: E402
from examples import get_preset, get_preset_list   # noqa: E402


# ──────────────────────────────────────────────────────────────
# 绘图工具
# ──────────────────────────────────────────────────────────────

def draw_waveform(data, color: str, title: str, ylabel: str = "") -> str:
    """用 matplotlib 绘制波形图，返回 base64 PNG."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(data, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.5, 1.8), dpi=100)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    x = np.arange(len(data))
    ax.plot(x, data, color=color, linewidth=1.0, alpha=0.9)
    ax.fill_between(x, data, alpha=0.08, color=color)

    ax.set_xlim(0, len(data))
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.set_title(title, color="#94a3b8", fontsize=10, pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, color="#64748b", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#2a3550")
    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def draw_tcm_bars(probabilities: dict, highlight_idx: int) -> str:
    """绘制中医体质概率条形图."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [
        "#10b981", "#f43f5e", "#f59e0b", "#8b5cf6",
        "#06b6d4", "#f97316", "#ec4899", "#14b8a6", "#6366f1",
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    bars = ax.barh(names, values, color=colors, height=0.55, zorder=3)
    # 高亮最高概率条
    for i, bar in enumerate(bars):
        if i == highlight_idx:
            bar.set_alpha(1.0)
            bar.set_edgecolor("#fff")
            bar.set_linewidth(1.5)
        else:
            bar.set_alpha(0.4)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", color="#64748b", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#2a3550")
    ax.grid(axis="x", color="#2a3550", alpha=0.5, zorder=0)

    # 标注最高值
    ax.text(
        values[highlight_idx] + 0.02,
        highlight_idx,
        f"{values[highlight_idx]:.1%}",
        va="center", color="#fff", fontsize=9, fontweight="bold"
    )

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def draw_mindfulness_ring(score: float) -> str:
    """绘制正念指数环形图."""
    import base64
    import io
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.set_aspect("equal")
    ax.axis("off")

    # 背景环
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#2a3550", linewidth=8, solid_capstyle="round")

    # 填充环
    end_angle = score * 2 * np.pi
    theta_fill = np.linspace(0, end_angle, 200)
    color = "#06b6d4" if score >= 0.6 else ("#f59e0b" if score >= 0.35 else "#f43f5e")
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=8, solid_capstyle="round")

    # 中心文字
    ax.text(0, 0.05, f"{score:.2f}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#e2e8f0")
    ax.text(0, -0.2, "正念指数", ha="center", va="center",
            fontsize=9, color="#64748b")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#111827", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def build_vitals_table(vitals: dict) -> str:
    """生成生命体征 HTML 表格."""
    rows = [
        ("💓 心率", f"{vitals['heart_rate']} bpm"),
        ("🩸 血氧", f"{vitals['spo2']}%"),
        ("🫁 呼吸率", f"{vitals['resp_rate']} 次/分"),
        ("🌡️ 体温", f"{vitals['temperature']}°C"),
    ]
    cells = "".join(
        f"<tr><td>{k}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
        for k, v in rows
    )
    return f"<table style='width:100%;border-collapse:collapse'>{cells}</table>"


def build_profile_table(profile: dict) -> str:
    """生成用户画像 HTML 表格."""
    rows = [
        ("年龄", f"{profile['age']} 岁"),
        ("性别", profile["gender"]),
        ("身高", f"{profile['height']} cm"),
        ("体重", f"{profile['weight']} kg"),
        ("BMI", f"{profile['bmi']:.1f}"),
        ("心率", f"{profile['heart_rate']} bpm"),
        ("血氧", f"{profile['spo2']}%"),
    ]
    cells = "".join(
        f"<tr><td>{k}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
        for k, v in rows
    )
    return f"<table style='width:100%;border-collapse:collapse'>{cells}</table>"


# ──────────────────────────────────────────────────────────────
# 管线示意图 HTML
# ──────────────────────────────────────────────────────────────

PIPELINE_HTML = """
<div style="display:flex;align-items:center;justify-content:center;gap:0;flex-wrap:wrap;padding:12px 0">
  <div style="text-align:center;padding:8px 6px;min-width:72px">
    <div style="width:44px;height:44px;border-radius:12px;background:#1a2236;border:2px solid #2a3550;
                display:flex;align-items:center;justify-content:center;font-size:20px;margin:0 auto 4px">🫀</div>
    <div style="font-size:10px;color:#64748b">生理信号</div>
  </div>
  <div style="color:#2a3550;font-size:16px;padding:0 2px">→</div>
  <div style="text-align:center;padding:8px 6px;min-width:72px">
    <div style="width:44px;height:44px;border-radius:12px;background:#1a2236;border:2px solid #2a3550;
                display:flex;align-items:center;justify-content:center;font-size:20px;margin:0 auto 4px">🧠</div>
    <div style="font-size:10px;color:#64748b">TCN 编码</div>
  </div>
  <div style="color:#2a3550;font-size:16px;padding:0 2px">→</div>
  <div style="text-align:center;padding:8px 6px;min-width:72px">
    <div style="width:44px;height:44px;border-radius:12px;background:#1a2236;border:2px solid #2a3550;
                display:flex;align-items:center;justify-content:center;font-size:20px;margin:0 auto 4px">🏮</div>
    <div style="font-size:10px;color:#64748b">FT 体质</div>
  </div>
  <div style="color:#2a3550;font-size:16px;padding:0 2px">→</div>
  <div style="text-align:center;padding:8px 6px;min-width:72px">
    <div style="width:44px;height:44px;border-radius:12px;background:#1a2236;border:2px solid #2a3550;
                display:flex;align-items:center;justify-content:center;font-size:20px;margin:0 auto 4px">🧬</div>
    <div style="font-size:10px;color:#64748b">EEG 正念</div>
  </div>
  <div style="color:#2a3550;font-size:16px;padding:0 2px">→</div>
  <div style="text-align:center;padding:8px 6px;min-width:72px">
    <div style="width:44px;height:44px;border-radius:12px;background:#1a2236;border:2px solid #06b6d4;
                display:flex;align-items:center;justify-content:center;font-size:20px;margin:0 auto 4px">🔗</div>
    <div style="font-size:10px;color:#06b6d4">统一融合</div>
  </div>
  <div style="color:#2a3550;font-size:16px;padding:0 2px">→</div>
  <div style="text-align:center;padding:8px 6px;min-width:72px">
    <div style="width:44px;height:44px;border-radius:12px;background:#1a2236;border:2px solid #10b981;
                display:flex;align-items:center;justify-content:center;font-size:20px;margin:0 auto 4px">🎯</div>
    <div style="font-size:10px;color:#10b981">按摩决策</div>
  </div>
</div>
"""


# ──────────────────────────────────────────────────────────────
# 推理主函数
# ──────────────────────────────────────────────────────────────

def run_inference(preset_key: str, progress=gr.Progress()) -> tuple:
    """对选定预设样本执行完整推理."""
    sample = get_preset(preset_key)
    manager = get_manager()
    result = manager.run_inference(sample)

    # ── 绘图 ──
    ecg_img = draw_waveform(sample["ecg"], "#06b6d4", "ECG 心电图", "mV")
    eda_img = draw_waveform(sample["eda"], "#f59e0b", "EDA 皮肤电导", "μS")
    eeg_img = draw_waveform(sample["eeg"], "#8b5cf6", "EEG 脑电", "μV")
    tcm_img = draw_tcm_bars(result["constitution"]["probabilities"],
                            result["constitution"]["index"])
    ring_img = draw_mindfulness_ring(result["neuro_repr"]["mindfulness_score"])

    # ── 推荐结果 HTML ──
    rec = result["recommendation"]
    prog = rec["program"]
    intensity = rec["intensity"]

    rec_html = f"""
    <div style="text-align:center;padding:16px">
      <div style="font-size:48px;margin-bottom:8px">{prog['icon']}</div>
      <div style="font-size:22px;font-weight:700">{prog['name']}
        <span style="font-weight:400;color:#64748b;font-size:14px">{prog['name_en']}</span></div>
      <div style="color:#94a3b8;font-size:13px;margin-top:4px">{prog['desc']}</div>
      <div style="margin-top:16px;display:inline-flex;align-items:center;gap:8px;
                  padding:8px 20px;background:#111827;border:2px solid #06b6d4;
                  border-radius:999px;font-size:16px;font-weight:600">
        {intensity['emoji']} {intensity['name']} ({intensity['pressure']})
      </div>
      <div style="margin-top:12px;display:flex;gap:16px;justify-content:center;flex-wrap:wrap">
        <span style="color:#94a3b8;font-size:13px">🧬 体质: <b style="color:#f59e0b">{rec['constitution']}</b></span>
        <span style="color:#94a3b8;font-size:13px">📊 置信度: <b style="color:#10b981">{rec['confidence']:.1%}</b></span>
        <span style="color:#94a3b8;font-size:13px">🧘 正念指数: <b style="color:#8b5cf6">{result['neuro_repr']['mindfulness_score']:.2f}</b></span>
      </div>
      <div style="margin-top:12px;font-size:12px;color:#64748b">
        适用手法: {', '.join(prog['techniques'])}
      </div>
    </div>
    """

    # ── 诊断特征 HTML ──
    tcm_input = sample["tcm"]
    tcm_input_html = f"""
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="color:#64748b;padding:4px 0">舌色评分</td>
          <td style="text-align:right;font-weight:600;color:#f59e0b">{tcm_input['tongue']:.1f}/10</td></tr>
      <tr><td style="color:#64748b;padding:4px 0">舌苔评分</td>
          <td style="text-align:right;font-weight:600;color:#f59e0b">{tcm_input['coating']:.1f}/10</td></tr>
      <tr><td style="color:#64748b;padding:4px 0">脉象评分</td>
          <td style="text-align:right;font-weight:600;color:#f59e0b">{tcm_input['pulse']:.1f}/10</td></tr>
      <tr><td style="color:#64748b;padding:4px 0">面诊评分</td>
          <td style="text-align:right;font-weight:600;color:#f59e0b">{tcm_input['face']:.1f}/10</td></tr>
    </table>
    """

    # ── 体征数据 ──
    vitals_html = build_vitals_table(sample["vitals"])

    # ── 用户画像 ──
    profile_html = build_profile_table(sample["profile"])

    # ── 动态表征详情 ──
    dyn_repr = result["dynamic_repr"]
    dyn_detail = f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <div style="flex:1;min-width:100px;background:#111827;padding:10px;border-radius:8px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:#06b6d4">128-D</div>
        <div style="font-size:10px;color:#64748b">动态表征向量</div>
      </div>
      <div style="flex:1;min-width:100px;background:#111827;padding:10px;border-radius:8px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:#10b981">Gate A ✅</div>
        <div style="font-size:10px;color:#64748b">TCM 条件调制</div>
      </div>
      <div style="flex:1;min-width:100px;background:#111827;padding:10px;border-radius:8px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:#f59e0b">Gate B ✅</div>
        <div style="font-size:10px;color:#64748b">质量校正</div>
      </div>
    </div>
    """

    # ── 神经表征详情 ──
    neuro_detail = f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
      <div style="flex:0 0 auto">
        <img src="{ring_img}" style="width:100px;height:100px" />
      </div>
      <div style="flex:1;min-width:120px">
        <div style="font-size:18px;font-weight:700;color:#8b5cf6">8-D</div>
        <div style="font-size:10px;color:#64748b">脑电神经表征</div>
        <div style="margin-top:8px;font-size:12px;color:#94a3b8">
          <b style="color:#8b5cf6">正念指数: {result['neuro_repr']['mindfulness_score']:.3f}</b><br/>
          <span style="color:#64748b">
            {'Alpha 节律占优 — 专注放松状态' if result['neuro_repr']['mindfulness_score'] >= 0.6 else
             '混合脑波 — 可提升正念训练' if result['neuro_repr']['mindfulness_score'] >= 0.35 else
             'Beta/Theta 偏高 — 建议正念干预'}
          </span>
        </div>
      </div>
    </div>
    """

    # ── 管线步骤状态 HTML ──
    steps_html = """
    <div id="pipeline-steps">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
        <div style="text-align:center;padding:6px 10px;background:rgba(16,185,129,.1);border-radius:8px">
          <div style="font-size:16px">🏮</div><div style="font-size:10px;color:#10b981">TCM 编码</div>
        </div>
        <div style="color:#10b981">→</div>
        <div style="text-align:center;padding:6px 10px;background:rgba(16,185,129,.1);border-radius:8px">
          <div style="font-size:16px">🧠</div><div style="font-size:10px;color:#10b981">动态编码</div>
        </div>
        <div style="color:#10b981">→</div>
        <div style="text-align:center;padding:6px 10px;background:rgba(16,185,129,.1);border-radius:8px">
          <div style="font-size:16px">🧬</div><div style="font-size:10px;color:#10b981">EEG 编码</div>
        </div>
        <div style="color:#10b981">→</div>
        <div style="text-align:center;padding:6px 10px;background:rgba(16,185,129,.1);border-radius:8px">
          <div style="font-size:16px">🔗</div><div style="font-size:10px;color:#10b981">融合决策</div>
        </div>
      </div>
    </div>
    """

    return (
        # 左侧输入面板
        sample["description"],
        profile_html,
        ecg_img, eda_img, eeg_img,
        tcm_input_html,
        vitals_html,
        f"<b>正念指数:</b> {sample['mindfulness']:.2f}",
        # 右侧输出面板
        steps_html,
        tcm_img,
        dyn_detail,
        result["constitution"]["name"],
        f"{result['constitution']['confidence']:.1%}",
        neuro_detail,
        rec_html,
    )


# ──────────────────────────────────────────────────────────────
# Gradio 界面构建
# ──────────────────────────────────────────────────────────────

def build_ui():
    manager = get_manager()
    presets = get_preset_list()

    with gr.Blocks(
        title="MulitiModal — 多模态按摩决策演示",
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
            font=["-apple-system", "PingFang SC", "Microsoft YaHei", "sans-serif"],
        ),
        css="""
        .gr-block { background: #111827 !important; }
        .gr-box { background: #111827 !important; border: 1px solid #2a3550 !important; border-radius: 12px !important; }
        .gr-form { background: #111827 !important; }
        .gr-input-text { background: #1a2236 !important; border: 1px solid #2a3550 !important; color: #e2e8f0 !important; }
        .gr-button { background: linear-gradient(135deg, #06b6d4, #8b5cf6) !important; }
        .gr-prose { color: #94a3b8 !important; }
        h1, h2, h3 { color: #e2e8f0 !important; }
        .gr-padded { padding: 16px !important; }
        """,
    ) as demo:
        # ── Header ──
        gr.HTML("""
        <div style="text-align:center;padding:24px 16px 8px;position:relative">
          <h1 style="font-size:32px;font-weight:800;background:linear-gradient(135deg,#06b6d4,#8b5cf6);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
            MulitiModal
          </h1>
          <p style="color:#94a3b8;font-size:14px;margin-top:6px">
            多模态人体状态感知 → 中医体质分析 → 智能按摩方案推荐
          </p>
        </div>
        """)

        # ── Pipeline diagram ──
        gr.HTML(f'<div style="max-width:900px;margin:0 auto">{PIPELINE_HTML}</div>')

        # ── 场景选择 ──
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=[(label, key) for key, label, _ in presets],
                value=presets[0][1],
                label="选择演示场景",
                scale=3,
            )
            run_btn = gr.Button("▶ 运行推理", variant="primary", scale=1)

        gr.Markdown("---")

        # ── 状态提示 ──
        status_md = gr.Markdown("", visible=False)

        # ── 左侧: 输入数据 ──
        with gr.Row():
            # 左列: 输入面板
            with gr.Column(scale=1):
                gr.HTML('<div style="font-size:14px;font-weight:600;color:#94a3b8;margin-bottom:8px">📥 输入数据</div>')

                description_md = gr.Markdown("")
                profile_md = gr.Markdown("")

                with gr.Accordion("🫀 生理信号波形 (ECG + EDA)", open=True):
                    ecg_img = gr.Image(label="ECG", show_label=False, height=130)
                    eda_img = gr.Image(label="EDA", show_label=False, height=130)

                with gr.Accordion("🧬 脑电波形 (EEG)", open=False):
                    eeg_img = gr.Image(label="EEG", show_label=False, height=130)

                with gr.Accordion("🏮 中医诊断特征", open=True):
                    tcm_input_md = gr.Markdown("")

                with gr.Accordion("📊 生命体征", open=False):
                    vitals_md = gr.Markdown("")

                with gr.Accordion("🧘 正念指数", open=False):
                    mindfulness_md = gr.Markdown("")

            # 右列: 输出面板
            with gr.Column(scale=1):
                gr.HTML('<div style="font-size:14px;font-weight:600;color:#94a3b8;margin-bottom:8px">📤 推理结果</div>')

                pipeline_steps_md = gr.HTML("")

                with gr.Accordion("🏮 Step 1: 中医体质编码 (FT-Transformer)", open=True):
                    tcm_output_md = gr.Markdown("")

                with gr.Accordion("🧠 Step 2: 动态表征 (TCN + Gate A/B)", open=True):
                    dyn_output_md = gr.HTML("")

                with gr.Accordion("🧬 Step 3: 神经表征 (EEG 编码器)", open=False):
                    neuro_output_md = gr.HTML("")

                with gr.Accordion("🎯 按摩决策结果", open=True):
                    recommendation_md = gr.HTML("")

        # ── 底部信息 ──
        gr.HTML(f"""
        <div style="text-align:center;padding:20px;color:#64748b;font-size:11px;
                    border-top:1px solid #2a3550;margin-top:24px">
          MulitiModal Demo · 模型权重: OPLRI + TCM (FT-Transformer) · 仅作技术演示用途<br/>
          OPLRI SHA-256: 89f75e66…e2f65ba &nbsp;|&nbsp;
          TCM SHA-256: b5c92665…e9422f0
        </div>
        """)

        # ── 事件绑定 ──
        def _do_infer(preset_key, progress=gr.Progress()):
            progress(0.1, desc="加载模型...")
            progress(0.3, desc="预处理输入数据...")
            result = run_inference(preset_key, progress)
            progress(0.9, desc="生成可视化...")
            progress(1.0, desc="完成")
            return (gr.update(visible=False),) + result

        run_btn.click(
            _do_infer,
            inputs=[preset_dropdown],
            outputs=[
                status_md,
                description_md, profile_md,
                ecg_img, eda_img, eeg_img,
                tcm_input_md, vitals_md, mindfulness_md,
                pipeline_steps_md, tcm_output_md, dyn_output_md,
                constitution_md := gr.Markdown(""),
                confidence_md := gr.Markdown(""),
                neuro_output_md, recommendation_md,
            ],
        )

        # 页面加载时自动运行第一个预设
        demo.load(
            _do_infer,
            inputs=[preset_dropdown],
            outputs=[
                status_md,
                description_md, profile_md,
                ecg_img, eda_img, eeg_img,
                tcm_input_md, vitals_md, mindfulness_md,
                pipeline_steps_md, tcm_output_md, dyn_output_md,
                constitution_md, confidence_md,
                neuro_output_md, recommendation_md,
            ],
        )

    return demo


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
