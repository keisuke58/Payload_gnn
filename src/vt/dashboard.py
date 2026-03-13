#!/usr/bin/env python3
"""
H3 Virtual Twin — Mission Dashboard

Orchestrator の結果を Plotly でインタラクティブ HTML ダッシュボードに出力。
Takram H3 FIP + NASA Open MCT のレイアウトを参考にした統合表示。

Usage:
    python -m src.vt.dashboard              # H3-22S
    python -m src.vt.dashboard H3-30S       # H3-30S
    python -m src.vt.dashboard H3-22S out.html
"""

import sys
import os
import math
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.vt.orchestrator import H3FlightOrchestrator
from src.vt.propulsion import G0


def build_dashboard(config: str = "H3-22S", output_path: str = None) -> str:
    """
    H3 ミッションダッシュボードを生成

    Args:
        config: H3 configuration ("H3-22S", "H3-24L", "H3-30S")
        output_path: HTML 出力パス (default: results/h3_dashboard_{config}.html)

    Returns:
        出力ファイルパス
    """
    # ── シミュレーション実行 ──
    print(f"Running {config} simulation...")
    orch = H3FlightOrchestrator(config, dt=0.5)
    result = orch.run_mission()

    t = result["t"]
    alt = result["altitude"] / 1e3       # [km]
    speed = result["speed"]               # [m/s]
    mach = result["mach"]
    mass = result["mass"] / 1e3           # [ton]
    q_dyn = result["q_dyn"] / 1e3         # [kPa]
    thrust = result["thrust"] / 1e3       # [kN]
    drag = result["drag"] / 1e3           # [kN]
    accel_G = result["accel_G"]
    gamma = result["gamma_deg"]
    pitch = result["pitch_deg"]
    T_nose = result["T_nose"]
    T_body = result["T_body"]
    q_stag = result["q_stag"] / 1e3       # [kW/m²]
    downrange = result["x"] / 1e3         # [km]
    gimbal = result["gimbal_pitch"]

    events = result["events"]

    # ── カラーパレット (Takram H3 FIP inspired) ──
    C_BG = "#0a0e1a"
    C_PANEL = "#111827"
    C_GRID = "#1e293b"
    C_TEXT = "#e2e8f0"
    C_ACCENT = "#38bdf8"
    C_THRUST = "#f97316"
    C_DRAG = "#ef4444"
    C_TEMP = "#fbbf24"
    C_MASS = "#a78bfa"
    C_ALT = "#34d399"
    C_SPEED = "#38bdf8"
    C_Q = "#f472b6"
    C_EVENT = "#94a3b8"

    # ── イベントアノテーション生成 ──
    event_annotations = []
    event_shapes = []
    for evt_t, evt_name, evt_desc in events:
        short = evt_name.replace("_", " ")
        event_annotations.append(dict(
            x=evt_t, y=1.02, xref="x", yref="paper",
            text=f"<b>{short}</b>", showarrow=True,
            arrowhead=2, arrowcolor=C_EVENT, arrowwidth=1,
            font=dict(size=9, color=C_EVENT),
            ax=0, ay=-25,
        ))
        event_shapes.append(dict(
            type="line", x0=evt_t, x1=evt_t, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=C_EVENT, width=1, dash="dot"),
        ))

    # ═══════════════════════════════════════════════════════
    # Figure 1: Main Flight Profile (4 rows)
    # ═══════════════════════════════════════════════════════
    fig = make_subplots(
        rows=4, cols=2,
        row_heights=[0.30, 0.25, 0.25, 0.20],
        column_widths=[0.55, 0.45],
        subplot_titles=[
            "Altitude & Downrange", "Velocity & Mach",
            "Thrust & Drag", "Dynamic Pressure (Max-Q)",
            "Temperature (Nose / Body)", "Mass & Acceleration",
            "Gimbal Angle", "Flight Path & Pitch Angle",
        ],
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # ── Row 1, Col 1: Altitude + Downrange ──
    fig.add_trace(go.Scatter(
        x=t, y=alt, name="Altitude [km]",
        line=dict(color=C_ALT, width=2.5),
        hovertemplate="t=%{x:.0f}s<br>h=%{y:.1f} km",
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=downrange, name="Downrange [km]",
        line=dict(color=C_ALT, width=1.5, dash="dash"),
        hovertemplate="t=%{x:.0f}s<br>x=%{y:.0f} km",
    ), row=1, col=1, secondary_y=True)

    # ── Row 1, Col 2: Velocity + Mach ──
    fig.add_trace(go.Scatter(
        x=t, y=speed, name="Velocity [m/s]",
        line=dict(color=C_SPEED, width=2.5),
        hovertemplate="t=%{x:.0f}s<br>V=%{y:.0f} m/s",
    ), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=mach, name="Mach",
        line=dict(color=C_SPEED, width=1.5, dash="dash"),
        hovertemplate="t=%{x:.0f}s<br>M=%{y:.1f}",
    ), row=1, col=2, secondary_y=True)

    # ── Row 2, Col 1: Thrust + Drag ──
    fig.add_trace(go.Scatter(
        x=t, y=thrust, name="Thrust [kN]",
        line=dict(color=C_THRUST, width=2.5),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.15)",
        hovertemplate="t=%{x:.0f}s<br>F=%{y:.0f} kN",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=drag, name="Drag [kN]",
        line=dict(color=C_DRAG, width=2),
        hovertemplate="t=%{x:.0f}s<br>D=%{y:.1f} kN",
    ), row=2, col=1)

    # ── Row 2, Col 2: Dynamic Pressure ──
    fig.add_trace(go.Scatter(
        x=t, y=q_dyn, name="q [kPa]",
        line=dict(color=C_Q, width=2.5),
        fill="tozeroy", fillcolor="rgba(244,114,182,0.15)",
        hovertemplate="t=%{x:.0f}s<br>q=%{y:.1f} kPa",
    ), row=2, col=2)
    # Max-Q marker
    i_maxq = np.argmax(q_dyn)
    fig.add_trace(go.Scatter(
        x=[t[i_maxq]], y=[q_dyn[i_maxq]],
        mode="markers+text",
        marker=dict(color=C_Q, size=12, symbol="star"),
        text=[f"Max-Q: {q_dyn[i_maxq]:.1f} kPa"],
        textposition="top right",
        textfont=dict(color=C_Q, size=11),
        name="Max-Q", showlegend=False,
    ), row=2, col=2)

    # ── Row 3, Col 1: Temperature ──
    fig.add_trace(go.Scatter(
        x=t, y=T_nose, name="T_nose [K]",
        line=dict(color=C_TEMP, width=2.5),
        hovertemplate="t=%{x:.0f}s<br>T_nose=%{y:.0f} K",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=T_body, name="T_body [K]",
        line=dict(color=C_TEMP, width=1.5, dash="dash"),
        hovertemplate="t=%{x:.0f}s<br>T_body=%{y:.0f} K",
    ), row=3, col=1)

    # ── Row 3, Col 2: Mass + Accel ──
    fig.add_trace(go.Scatter(
        x=t, y=mass, name="Mass [ton]",
        line=dict(color=C_MASS, width=2.5),
        hovertemplate="t=%{x:.0f}s<br>m=%{y:.1f} ton",
    ), row=3, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=accel_G, name="Accel [G]",
        line=dict(color="#fb923c", width=1.5, dash="dash"),
        hovertemplate="t=%{x:.0f}s<br>a=%{y:.1f} G",
    ), row=3, col=2, secondary_y=True)

    # ── Row 4, Col 1: Gimbal ──
    fig.add_trace(go.Scatter(
        x=t, y=gimbal, name="Gimbal [°]",
        line=dict(color="#60a5fa", width=2),
        hovertemplate="t=%{x:.0f}s<br>δ=%{y:.2f}°",
    ), row=4, col=1)

    # ── Row 4, Col 2: Flight Path + Pitch ──
    fig.add_trace(go.Scatter(
        x=t, y=gamma, name="γ (flight path) [°]",
        line=dict(color="#4ade80", width=2),
        hovertemplate="t=%{x:.0f}s<br>γ=%{y:.1f}°",
    ), row=4, col=2)
    fig.add_trace(go.Scatter(
        x=t, y=pitch, name="θ (pitch) [°]",
        line=dict(color="#818cf8", width=1.5, dash="dash"),
        hovertemplate="t=%{x:.0f}s<br>θ=%{y:.1f}°",
    ), row=4, col=2)

    # ── イベントライン (全サブプロットに) ──
    for shape in event_shapes:
        fig.add_shape(shape)

    # ── レイアウト ──
    summary = orch.mission_summary(result)
    i_end = len(t) - 1

    # KPI テキスト
    kpi_text = (
        f"<b>H3 Virtual Twin — {config}</b><br>"
        f"Liftoff: {mass[0]:.0f} ton | "
        f"Max-Q: {q_dyn[i_maxq]:.1f} kPa (T+{t[i_maxq]:.0f}s) | "
        f"V_SECO: {speed[i_end]:.0f} m/s | "
        f"h_SECO: {alt[i_end]:.0f} km | "
        f"Max G: {np.max(accel_G):.1f} | "
        f"T_nose_max: {np.max(T_nose):.0f} K"
    )

    fig.update_layout(
        title=dict(
            text=kpi_text,
            font=dict(size=14, color=C_ACCENT),
            x=0.5, xanchor="center",
        ),
        height=1200,
        width=1400,
        template="plotly_dark",
        paper_bgcolor=C_BG,
        plot_bgcolor=C_PANEL,
        font=dict(family="Inter, monospace", size=11, color=C_TEXT),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.05,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
        # Event annotations on top
        annotations=event_annotations,
    )

    # Grid styling
    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(
                gridcolor=C_GRID, zeroline=False,
                row=i, col=j,
            )
            fig.update_yaxes(
                gridcolor=C_GRID, zeroline=False,
                row=i, col=j,
            )

    # Y-axis labels
    fig.update_yaxes(title_text="Altitude [km]", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Downrange [km]", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Velocity [m/s]", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Mach", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Force [kN]", row=2, col=1)
    fig.update_yaxes(title_text="q [kPa]", row=2, col=2)
    fig.update_yaxes(title_text="Temperature [K]", row=3, col=1)
    fig.update_yaxes(title_text="Mass [ton]", row=3, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Accel [G]", row=3, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Gimbal [°]", row=4, col=1)
    fig.update_yaxes(title_text="Angle [°]", row=4, col=2)

    # X-axis: time
    for j in range(1, 3):
        fig.update_xaxes(title_text="Time [s]", row=4, col=j)

    # ═══════════════════════════════════════════════════════
    # Figure 2: Trajectory Map (altitude vs downrange)
    # ═══════════════════════════════════════════════════════
    fig2 = go.Figure()

    # Color by Mach number
    fig2.add_trace(go.Scatter(
        x=downrange, y=alt,
        mode="lines+markers",
        marker=dict(
            size=4, color=mach,
            colorscale="Turbo", showscale=True,
            colorbar=dict(title="Mach", x=1.02),
        ),
        line=dict(color="rgba(56,189,248,0.4)", width=1),
        hovertemplate=(
            "x=%{x:.0f} km<br>h=%{y:.1f} km<br>"
            "M=%{marker.color:.1f}"
        ),
        name="Trajectory",
    ))

    # Event markers
    for evt_t, evt_name, _ in events:
        idx = min(int(evt_t / orch.dt), i_end)
        fig2.add_trace(go.Scatter(
            x=[downrange[idx]], y=[alt[idx]],
            mode="markers+text",
            marker=dict(size=10, color="white", symbol="diamond",
                        line=dict(color=C_ACCENT, width=2)),
            text=[evt_name.replace("_", " ")],
            textposition="top right",
            textfont=dict(size=10, color=C_TEXT),
            showlegend=False,
        ))

    fig2.update_layout(
        title=dict(
            text=f"<b>{config} — Trajectory Profile</b>",
            font=dict(size=16, color=C_ACCENT),
        ),
        xaxis_title="Downrange [km]",
        yaxis_title="Altitude [km]",
        height=500, width=1400,
        template="plotly_dark",
        paper_bgcolor=C_BG, plot_bgcolor=C_PANEL,
        font=dict(family="Inter, monospace", size=11, color=C_TEXT),
    )
    fig2.update_xaxes(gridcolor=C_GRID)
    fig2.update_yaxes(gridcolor=C_GRID)

    # ═══════════════════════════════════════════════════════
    # Figure 3: Events Timeline
    # ═══════════════════════════════════════════════════════
    fig3 = go.Figure()

    event_names = [e[1].replace("_", " ") for e in events]
    event_times = [e[0] for e in events]
    event_alts = []
    event_speeds = []
    for evt_t, _, _ in events:
        idx = min(int(evt_t / orch.dt), i_end)
        event_alts.append(f"{alt[idx]:.0f} km")
        event_speeds.append(f"{speed[idx]:.0f} m/s")

    colors = ["#f97316", "#ef4444", "#a78bfa", "#38bdf8",
              "#34d399", "#fbbf24", "#f472b6"]

    fig3.add_trace(go.Bar(
        x=event_times, y=[1]*len(events),
        text=[f"<b>{n}</b><br>{a}<br>{v}"
              for n, a, v in zip(event_names, event_alts, event_speeds)],
        textposition="outside",
        marker_color=colors[:len(events)],
        width=8,
        hovertemplate="T+%{x:.0f}s<br>%{text}",
        showlegend=False,
    ))

    fig3.update_layout(
        title=dict(
            text=f"<b>{config} — Flight Events Timeline</b>",
            font=dict(size=16, color=C_ACCENT),
        ),
        xaxis_title="Mission Time [s]",
        height=250, width=1400,
        template="plotly_dark",
        paper_bgcolor=C_BG, plot_bgcolor=C_PANEL,
        font=dict(family="Inter, monospace", size=11, color=C_TEXT),
        yaxis=dict(visible=False),
        xaxis=dict(gridcolor=C_GRID),
        bargap=0.5,
    )

    # ── HTML 出力 ──
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        output_path = f"results/h3_dashboard_{config}.html"

    # Combine all figures into single HTML
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>H3 Virtual Twin Dashboard — {config}</title>
<style>
  body {{
    margin: 0; padding: 20px;
    background: {C_BG};
    color: {C_TEXT};
    font-family: 'Inter', 'Menlo', monospace;
  }}
  .header {{
    text-align: center; padding: 20px 0 10px;
  }}
  .header h1 {{
    color: {C_ACCENT}; font-size: 28px; margin: 0;
    letter-spacing: 2px;
  }}
  .header p {{
    color: #64748b; font-size: 14px; margin: 5px 0;
  }}
  .kpi-bar {{
    display: flex; justify-content: center; gap: 30px;
    padding: 15px; margin: 10px auto;
    max-width: 1400px;
    background: {C_PANEL};
    border: 1px solid #1e293b;
    border-radius: 8px;
  }}
  .kpi {{
    text-align: center;
  }}
  .kpi .value {{
    font-size: 24px; font-weight: bold;
  }}
  .kpi .label {{
    font-size: 11px; color: #64748b; text-transform: uppercase;
  }}
  .section {{ max-width: 1440px; margin: 0 auto; }}
  .footer {{
    text-align: center; padding: 20px;
    color: #475569; font-size: 12px;
  }}
</style>
</head>
<body>
<div class="header">
  <h1>H3 VIRTUAL TWIN</h1>
  <p>Flight Simulation Dashboard — {config} Configuration</p>
</div>
<div class="kpi-bar">
  <div class="kpi">
    <div class="value" style="color:{C_MASS}">{mass[0]:.0f} t</div>
    <div class="label">Liftoff Mass</div>
  </div>
  <div class="kpi">
    <div class="value" style="color:{C_Q}">{q_dyn[i_maxq]:.1f} kPa</div>
    <div class="label">Max-Q (T+{t[i_maxq]:.0f}s)</div>
  </div>
  <div class="kpi">
    <div class="value" style="color:{C_SPEED}">{speed[i_end]:.0f} m/s</div>
    <div class="label">V at SECO</div>
  </div>
  <div class="kpi">
    <div class="value" style="color:{C_ALT}">{alt[i_end]:.0f} km</div>
    <div class="label">h at SECO</div>
  </div>
  <div class="kpi">
    <div class="value" style="color:{C_THRUST}">{np.max(accel_G):.1f} G</div>
    <div class="label">Max Accel</div>
  </div>
  <div class="kpi">
    <div class="value" style="color:{C_TEMP}">{np.max(T_nose):.0f} K</div>
    <div class="label">Max T (nose)</div>
  </div>
  <div class="kpi">
    <div class="value" style="color:{C_MASS}">{mass[i_end]:.1f} t</div>
    <div class="label">Final Mass</div>
  </div>
</div>
<div class="section">
""")

    html_parts.append(fig3.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html_parts.append(f"""
</div>
<div class="footer">
  H3 Virtual Twin — Phase A+B Integrated Simulation | {config}<br>
  Propulsion (LE-9/SRB-3/LE-5B-3) + Aerodynamics + Aerothermal + Attitude Control + Flight Orchestrator<br>
  src/vt/ — 6 modules, ~3,500 lines
</div>
</body>
</html>""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"\n=> Dashboard saved: {output_path}")
    print(f"   Open in browser to view.")
    return output_path


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "H3-22S"
    output = sys.argv[2] if len(sys.argv) > 2 else None
    build_dashboard(config, output)
