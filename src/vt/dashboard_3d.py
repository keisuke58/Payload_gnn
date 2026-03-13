#!/usr/bin/env python3
"""
H3 Virtual Twin — 3D CesiumJS + Plotly Dashboard (Phase D)

Generates a single HTML file with:
  - CesiumJS 3D globe showing trajectory + MC dispersion corridor
  - Plotly flight profile panels (alt, vel, mach, q, temp, mass)
  - Timeline playback with event markers
  - KPI summary bar

Usage:
    python -m src.vt.dashboard_3d              # Generate from saved .npz
    python -m src.vt.dashboard_3d --run        # Run sim + dashboard
    python -m src.vt.dashboard_3d --mc 30      # MC + dashboard
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def generate_dashboard_html(telemetry: dict, events: list,
                            mc_telemetry: list = None,
                            mc_stats: dict = None,
                            config: str = "H3-22S") -> str:
    """
    Generate CesiumJS + Plotly integrated dashboard HTML.

    Args:
        telemetry: dict of numpy arrays from 6DOF
        events: list of event dicts
        mc_telemetry: optional list of telemetry dicts from MC runs
        mc_stats: optional MC statistics dict
        config: rocket configuration name
    """
    t = telemetry["t"]
    lat = telemetry["lat"]
    lon = telemetry["lon"]
    alt = telemetry["alt"]
    speed = telemetry["speed"]
    mach = telemetry["mach"]
    q_dyn = telemetry["q_dyn"]
    accel_g = telemetry["accel_g"]
    mass = telemetry["mass"]
    thrust = telemetry["thrust"]
    drag = telemetry["drag"]
    gamma = telemetry["gamma"]
    pitch = telemetry["pitch"]
    T_nose = telemetry["T_nose"]
    v_inertial = telemetry["v_inertial"]
    downrange = telemetry["downrange"]
    phase = telemetry.get("phase", np.array([""] * len(t)))

    n = len(t)

    # ── Build CZML (CesiumJS timeline format) ──
    czml = _build_czml(t, lat, lon, alt, mach, events, config)

    # ── Build MC corridor points for CesiumJS ──
    mc_corridor_js = "null"
    if mc_telemetry and len(mc_telemetry) > 0:
        mc_corridor_js = _build_mc_corridor(mc_telemetry)

    # ── Build Plotly data as JSON ──
    plotly_data = _build_plotly_data(t, alt, speed, mach, q_dyn, accel_g,
                                     mass, thrust, drag, gamma, pitch,
                                     T_nose, v_inertial, downrange, events)

    # ── KPIs ──
    kpi = _compute_kpis(telemetry, events, mc_stats)

    # ── Event markers for timeline ──
    events_js = json.dumps(events, default=str)

    # ── Assemble HTML ──
    html = _assemble_html(czml, mc_corridor_js, plotly_data, kpi, events_js,
                          config, n, t[-1])
    return html


def _build_czml(t, lat, lon, alt, mach, events, config):
    """Build CZML document for CesiumJS."""
    t0_iso = "2025-03-01T10:00:00Z"  # Fictional launch time

    # Trajectory path positions [time, lon, lat, alt, ...]
    positions = []
    # Sample every few points for performance
    step = max(1, len(t) // 500)
    for i in range(0, len(t), step):
        positions.extend([float(t[i]), float(lon[i]), float(lat[i]), float(alt[i])])
    # Ensure last point
    if len(t) - 1 not in range(0, len(t), step):
        positions.extend([float(t[-1]), float(lon[-1]), float(lat[-1]), float(alt[-1])])

    # Color by Mach: subsonic=blue, transonic=green, supersonic=orange, hyper=red
    def mach_color(m):
        if m < 0.8:
            return [60, 130, 255, 255]
        elif m < 1.2:
            return [80, 220, 100, 255]
        elif m < 5:
            return [255, 165, 0, 255]
        else:
            return [255, 50, 50, 255]

    # Path color intervals
    color_intervals = []
    for i in range(0, len(t), step):
        c = mach_color(float(mach[i]))
        color_intervals.append({"interval": f"{t0_iso}/{t0_iso}", "rgba": c})

    czml_doc = [
        {
            "id": "document",
            "name": f"{config} 6DOF Trajectory",
            "version": "1.0",
            "clock": {
                "interval": f"{t0_iso}/{t0_iso}",
                "currentTime": t0_iso,
                "multiplier": 10,
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER"
            }
        },
        {
            "id": "trajectory",
            "name": f"{config} Flight Path",
            "position": {
                "epoch": t0_iso,
                "cartographicDegrees": positions,
            },
            "path": {
                "material": {
                    "solidColor": {
                        "color": {"rgba": [255, 140, 0, 200]}
                    }
                },
                "width": 3,
                "leadTime": 0,
                "trailTime": 1e10,
            },
            "point": {
                "pixelSize": 8,
                "color": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [255, 100, 0, 255]},
                "outlineWidth": 2,
            },
            "label": {
                "text": config,
                "font": "12pt monospace",
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 200]},
                "outlineWidth": 2,
                "pixelOffset": {"cartesian2": [10, -10]},
                "style": "FILL_AND_OUTLINE"
            }
        }
    ]

    # Event markers
    for ev in events:
        idx = int(np.argmin(np.abs(t - ev["t"])))
        czml_doc.append({
            "id": f"event_{ev['name']}",
            "name": ev["name"].replace("_", " "),
            "position": {
                "cartographicDegrees": [float(lon[idx]), float(lat[idx]), float(alt[idx])]
            },
            "point": {
                "pixelSize": 10,
                "color": {"rgba": [255, 255, 0, 255]},
                "outlineColor": {"rgba": [255, 0, 0, 255]},
                "outlineWidth": 2,
            },
            "label": {
                "text": ev["name"].replace("_", " "),
                "font": "11pt monospace",
                "fillColor": {"rgba": [255, 255, 100, 255]},
                "pixelOffset": {"cartesian2": [12, 0]},
                "style": "FILL"
            }
        })

    return json.dumps(czml_doc)


def _build_mc_corridor(mc_telemetry):
    """Build MC dispersion corridor as polyline positions."""
    # Find common time grid
    all_lats = []
    all_lons = []
    all_alts = []

    for telem in mc_telemetry:
        if "lat" in telem and len(telem["lat"]) > 10:
            all_lats.append(telem["lat"])
            all_lons.append(telem["lon"])
            all_alts.append(telem["alt"])

    if not all_lats:
        return "null"

    # Sample to common length
    n_min = min(len(l) for l in all_lats)
    step = max(1, n_min // 200)
    indices = list(range(0, n_min, step))

    runs_data = []
    for i in range(len(all_lats)):
        run_pts = []
        for idx in indices:
            if idx < len(all_lats[i]):
                run_pts.append([
                    float(all_lons[i][idx]),
                    float(all_lats[i][idx]),
                    float(all_alts[i][idx])
                ])
        runs_data.append(run_pts)

    return json.dumps(runs_data)


def _build_plotly_data(t, alt, speed, mach, q_dyn, accel_g, mass,
                       thrust, drag, gamma, pitch, T_nose, v_inertial,
                       downrange, events):
    """Build Plotly trace data as JSON."""
    step = max(1, len(t) // 1000)
    idx = list(range(0, len(t), step))

    data = {
        "t": [float(t[i]) for i in idx],
        "alt_km": [float(alt[i]) / 1000 for i in idx],
        "speed_kms": [float(speed[i]) / 1000 for i in idx],
        "mach": [float(mach[i]) for i in idx],
        "q_kpa": [float(q_dyn[i]) / 1000 for i in idx],
        "accel_g": [float(accel_g[i]) for i in idx],
        "mass_t": [float(mass[i]) / 1000 for i in idx],
        "thrust_kn": [float(thrust[i]) / 1000 for i in idx],
        "drag_kn": [float(drag[i]) / 1000 for i in idx],
        "gamma_deg": [float(gamma[i]) for i in idx],
        "pitch_deg": [float(pitch[i]) for i in idx],
        "T_nose": [float(T_nose[i]) for i in idx],
        "v_inertial_kms": [float(v_inertial[i]) / 1000 for i in idx],
        "downrange_km": [float(downrange[i]) for i in idx],
        "events": [
            {"name": e["name"], "t": float(e["t"]),
             "alt_km": float(e["alt"]) / 1000,
             "v_ms": float(e["speed"])}
            for e in events
        ]
    }
    return json.dumps(data)


def _compute_kpis(telem, events, mc_stats=None):
    """Compute KPI values for the header bar."""
    t = telem["t"]
    kpis = {
        "liftoff_mass": f"{telem['mass'][0]/1000:.0f} t",
        "max_q": f"{np.max(telem['q_dyn'])/1000:.1f} kPa",
        "max_g": f"{np.max(telem['accel_g']):.1f} G",
        "v_seco": f"{telem['speed'][-1]/1000:.2f} km/s",
        "vi_seco": f"{telem['v_inertial'][-1]/1000:.2f} km/s",
        "h_seco": f"{telem['alt'][-1]/1000:.0f} km",
        "max_T": f"{np.max(telem['T_nose']):.0f} K",
        "final_mass": f"{telem['mass'][-1]/1000:.1f} t",
    }

    # Add MC stats if available
    if mc_stats:
        if "h_seco" in mc_stats:
            s = mc_stats["h_seco"]
            kpis["h_seco_mc"] = f"{s['mean']/1000:.0f} ± {s['std']/1000:.0f} km"
        if "v_seco" in mc_stats:
            s = mc_stats["v_seco"]
            kpis["v_seco_mc"] = f"{s['mean']/1000:.2f} ± {s['std']/1000:.3f} km/s"

    return json.dumps(kpis)


def _assemble_html(czml_js, mc_corridor_js, plotly_data_js, kpi_js,
                   events_js, config, n_points, t_end):
    """Assemble the full HTML dashboard."""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{config} Virtual Twin — 6DOF Dashboard</title>
<script src="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/1.95/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0e1a;
    color: #e0e6f0;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    overflow-x: hidden;
  }}
  .header {{
    background: linear-gradient(90deg, #0f1629 0%, #1a2744 100%);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #2a3a5c;
  }}
  .header h1 {{
    font-size: 18px;
    font-weight: 600;
    color: #ff8c00;
    letter-spacing: 1px;
  }}
  .header .subtitle {{
    font-size: 11px;
    color: #6b7b9e;
    margin-left: 12px;
  }}
  .kpi-bar {{
    background: #111827;
    display: flex;
    gap: 2px;
    padding: 8px 12px;
    overflow-x: auto;
    border-bottom: 1px solid #1e293b;
  }}
  .kpi-item {{
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 110px;
    text-align: center;
  }}
  .kpi-label {{
    font-size: 9px;
    color: #6b7b9e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .kpi-value {{
    font-size: 14px;
    font-weight: 700;
    color: #60a5fa;
    margin-top: 2px;
  }}
  .kpi-value.warn {{ color: #f59e0b; }}
  .kpi-value.ok {{ color: #34d399; }}
  .kpi-mc {{
    font-size: 9px;
    color: #8b8fa3;
    margin-top: 1px;
  }}
  .main-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 50vh auto;
    gap: 0;
    min-height: calc(100vh - 100px);
  }}
  #cesiumContainer {{
    grid-column: 1;
    grid-row: 1;
    border-right: 1px solid #1e293b;
    border-bottom: 1px solid #1e293b;
  }}
  #plotly-main {{
    grid-column: 2;
    grid-row: 1;
    border-bottom: 1px solid #1e293b;
  }}
  #plotly-bottom {{
    grid-column: 1 / -1;
    grid-row: 2;
    padding: 4px;
  }}
  .event-bar {{
    background: #0f172a;
    padding: 6px 16px;
    display: flex;
    gap: 12px;
    align-items: center;
    border-bottom: 1px solid #1e293b;
    overflow-x: auto;
    flex-wrap: nowrap;
  }}
  .event-chip {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 3px;
    font-size: 10px;
    white-space: nowrap;
    border: 1px solid;
  }}
  .event-chip.srb {{ background: #1e1a00; border-color: #f59e0b; color: #fbbf24; }}
  .event-chip.sep {{ background: #001a1e; border-color: #06b6d4; color: #22d3ee; }}
  .event-chip.burn {{ background: #1a0000; border-color: #ef4444; color: #f87171; }}
  .event-chip.coast {{ background: #001a00; border-color: #22c55e; color: #4ade80; }}

  /* CesiumJS overrides */
  .cesium-viewer-bottom {{ display: none !important; }}
  .cesium-viewer .cesium-widget-credits {{ display: none !important; }}
</style>
</head>
<body>

<div class="header">
  <div style="display:flex;align-items:baseline;">
    <h1>JAXA {config} VIRTUAL TWIN</h1>
    <span class="subtitle">6DOF Trajectory + Monte Carlo Dispersion | {n_points} samples | T+{t_end:.0f}s</span>
  </div>
  <div style="font-size:10px;color:#4b5563;">Phase C/D — Quaternion 6DOF + WGS84/J2 + CesiumJS</div>
</div>

<div class="kpi-bar" id="kpiBar"></div>

<div class="event-bar" id="eventBar"></div>

<div class="main-grid">
  <div id="cesiumContainer"></div>
  <div id="plotly-main"></div>
  <div id="plotly-bottom"></div>
</div>

<script>
// Disable Cesium Ion completely — we use OSM tiles only
Cesium.Ion.defaultAccessToken = undefined;

// ── Data ──
const czmlData = {czml_js};
const mcCorridorData = {mc_corridor_js};
const plotlyData = {plotly_data_js};
const kpiData = {kpi_js};
const eventsData = {events_js};

// ── KPI Bar ──
(function() {{
  const bar = document.getElementById('kpiBar');
  const items = [
    ['Liftoff', kpiData.liftoff_mass, ''],
    ['Max-Q', kpiData.max_q, 'warn'],
    ['Max-G', kpiData.max_g, 'warn'],
    ['V_SECO', kpiData.v_seco, 'ok'],
    ['V_inertial', kpiData.vi_seco, 'ok'],
    ['h_SECO', kpiData.h_seco, ''],
    ['T_max', kpiData.max_T, 'warn'],
    ['Final Mass', kpiData.final_mass, ''],
  ];
  if (kpiData.h_seco_mc) items.push(['h_SECO (MC)', kpiData.h_seco_mc, '']);
  if (kpiData.v_seco_mc) items.push(['V_SECO (MC)', kpiData.v_seco_mc, '']);

  items.forEach(([label, value, cls]) => {{
    const d = document.createElement('div');
    d.className = 'kpi-item';
    d.innerHTML = `<div class="kpi-label">${{label}}</div><div class="kpi-value ${{cls}}">${{value}}</div>`;
    bar.appendChild(d);
  }});
}})();

// ── Event Bar ──
(function() {{
  const bar = document.getElementById('eventBar');
  const colorMap = {{
    'SRB_BURNOUT': 'srb', 'SRB_SEP': 'srb',
    'FAIRING_SEP': 'sep', 'STAGE_SEP': 'sep',
    'MECO': 'burn', 'S2_IGNITION': 'burn', 'SECO': 'coast',
  }};
  eventsData.forEach(ev => {{
    const d = document.createElement('div');
    const cls = colorMap[ev.name] || 'sep';
    d.className = `event-chip ${{cls}}`;
    const hkm = (ev.alt / 1000).toFixed(1);
    const vkms = (ev.speed / 1000).toFixed(1);
    d.innerHTML = `T+${{ev.t.toFixed(0)}}s <b>${{ev.name.replace(/_/g,' ')}}</b> h=${{hkm}}km V=${{vkms}}km/s`;
    bar.appendChild(d);
  }});
}})();

// ── CesiumJS 3D Globe (no Ion token required) ──
// Trajectory data: lon, lat, alt arrays
const trajData = (function() {{
  const raw = czmlData;
  // Extract positions from CZML entity
  const entity = raw.find(e => e.id === 'trajectory');
  if (!entity || !entity.position) return null;
  const pts = entity.position.cartographicDegrees;
  const coords = [];
  for (let i = 0; i < pts.length; i += 4) {{
    coords.push({{ t: pts[i], lon: pts[i+1], lat: pts[i+2], alt: pts[i+3] }});
  }}
  return coords;
}})();

// Initialize viewer — CesiumJS 1.95, no Ion token needed
const viewer = new Cesium.Viewer('cesiumContainer', {{
  imageryProvider: new Cesium.OpenStreetMapImageryProvider({{
    url: 'https://tile.openstreetmap.org/'
  }}),
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  baseLayerPicker: false,
  timeline: false,
  animation: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  fullscreenButton: false,
  geocoder: false,
  infoBox: false,
  selectionIndicator: false,
  shadows: false,
  shouldAnimate: false,
  requestRenderMode: true,
  maximumRenderTimeChange: Infinity,
}});

// Dark atmosphere
viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#0a0e1a');
viewer.scene.globe.enableLighting = false;
viewer.scene.fog.enabled = false;
viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#0a1628');

// ── Launch site marker ──
viewer.entities.add({{
  position: Cesium.Cartesian3.fromDegrees(130.9751, 30.4009, 20),
  point: {{ pixelSize: 12, color: Cesium.Color.LIME, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 }},
  label: {{
    text: 'Tanegashima',
    font: '12px monospace',
    fillColor: Cesium.Color.LIME,
    outlineColor: Cesium.Color.BLACK,
    outlineWidth: 2,
    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
    pixelOffset: new Cesium.Cartesian2(0, -20),
    disableDepthTestDistance: Number.POSITIVE_INFINITY,
  }},
}});

// ── Draw trajectory polyline segments (color by Mach) ──
if (trajData && trajData.length > 1) {{
  // Mach-based color function
  function machColor(m) {{
    if (m < 0.8) return Cesium.Color.fromCssColorString('#3c82ff');     // Subsonic: blue
    if (m < 1.2) return Cesium.Color.fromCssColorString('#50dc64');     // Transonic: green
    if (m < 5.0) return Cesium.Color.fromCssColorString('#ffa500');     // Supersonic: orange
    return Cesium.Color.fromCssColorString('#ff3232');                   // Hypersonic: red
  }}

  // Build positions array
  const allPositions = [];
  trajData.forEach(pt => {{
    allPositions.push(Cesium.Cartesian3.fromDegrees(pt.lon, pt.lat, pt.alt));
  }});

  // Draw segments in color groups
  let segStart = 0;
  let prevColorKey = -1;
  const mData = plotlyData.mach;
  const step = Math.max(1, Math.floor(mData.length / trajData.length));

  for (let i = 0; i < trajData.length; i++) {{
    const mIdx = Math.min(i * step, mData.length - 1);
    const m = mData[mIdx] || 0;
    const colorKey = m < 0.8 ? 0 : m < 1.2 ? 1 : m < 5 ? 2 : 3;

    if (colorKey !== prevColorKey && i > 0) {{
      // End previous segment, start new one
      const segPositions = allPositions.slice(segStart, i + 1);
      if (segPositions.length >= 2) {{
        const mPrev = mData[Math.min(segStart * step, mData.length - 1)] || 0;
        viewer.entities.add({{
          polyline: {{
            positions: segPositions,
            width: 3,
            material: new Cesium.ColorMaterialProperty(machColor(mPrev).withAlpha(0.9)),
            clampToGround: false,
          }}
        }});
      }}
      segStart = i;
    }}
    prevColorKey = colorKey;
  }}
  // Last segment
  const lastSeg = allPositions.slice(segStart);
  if (lastSeg.length >= 2) {{
    const mLast = mData[Math.min(segStart * step, mData.length - 1)] || 0;
    viewer.entities.add({{
      polyline: {{
        positions: lastSeg,
        width: 3,
        material: new Cesium.ColorMaterialProperty(machColor(mLast).withAlpha(0.9)),
        clampToGround: false,
      }}
    }});
  }}

  // ── Drop lines to ground (altitude visualization) ──
  const dropEvery = Math.max(1, Math.floor(trajData.length / 30));
  for (let i = 0; i < trajData.length; i += dropEvery) {{
    const pt = trajData[i];
    if (pt.alt > 5000) {{
      viewer.entities.add({{
        polyline: {{
          positions: [
            Cesium.Cartesian3.fromDegrees(pt.lon, pt.lat, pt.alt),
            Cesium.Cartesian3.fromDegrees(pt.lon, pt.lat, 0),
          ],
          width: 1,
          material: new Cesium.ColorMaterialProperty(
            Cesium.Color.fromCssColorString('rgba(100, 160, 255, 0.15)')
          ),
          clampToGround: false,
        }}
      }});
    }}
  }}
}}

// ── Event markers on globe ──
const eventColors = {{
  'SRB_BURNOUT': Cesium.Color.YELLOW, 'SRB_SEP': Cesium.Color.ORANGE,
  'FAIRING_SEP': Cesium.Color.CYAN, 'MECO': Cesium.Color.RED,
  'STAGE_SEP': Cesium.Color.DEEPSKYBLUE, 'S2_IGNITION': Cesium.Color.ORANGERED,
  'SECO': Cesium.Color.LIME,
}};
eventsData.forEach(ev => {{
  // Find nearest trajectory point
  let nearIdx = 0;
  if (trajData) {{
    let minDt = Infinity;
    trajData.forEach((pt, i) => {{
      const d = Math.abs(pt.t - ev.t);
      if (d < minDt) {{ minDt = d; nearIdx = i; }}
    }});
  }}
  const pt = trajData ? trajData[nearIdx] : null;
  if (!pt) return;

  const col = eventColors[ev.name] || Cesium.Color.WHITE;
  viewer.entities.add({{
    position: Cesium.Cartesian3.fromDegrees(pt.lon, pt.lat, pt.alt),
    point: {{ pixelSize: 10, color: col, outlineColor: Cesium.Color.WHITE, outlineWidth: 1 }},
    label: {{
      text: ev.name.replace(/_/g, ' ') + '\\nT+' + ev.t.toFixed(0) + 's',
      font: '10px monospace',
      fillColor: col,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      pixelOffset: new Cesium.Cartesian2(12, 0),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    }},
  }});
}});

// ── MC dispersion corridor ──
if (mcCorridorData && mcCorridorData.length > 0) {{
  mcCorridorData.forEach((run, i) => {{
    if (run.length < 2) return;
    const positions = run.map(pt =>
      Cesium.Cartesian3.fromDegrees(pt[0], pt[1], pt[2])
    );
    viewer.entities.add({{
      polyline: {{
        positions: positions,
        width: 1,
        material: new Cesium.ColorMaterialProperty(
          Cesium.Color.fromCssColorString('rgba(100, 170, 255, 0.12)')
        ),
        clampToGround: false,
      }}
    }});
  }});
}}

// ── Mach legend overlay ──
(function() {{
  const legend = document.createElement('div');
  legend.style.cssText = 'position:absolute;bottom:8px;left:8px;background:rgba(10,14,26,0.85);' +
    'border:1px solid #1e293b;border-radius:4px;padding:6px 10px;font-size:10px;z-index:100;';
  legend.innerHTML = '<div style="color:#9ca3af;margin-bottom:3px;font-weight:600">Mach</div>' +
    '<div><span style="color:#3c82ff">\\u25A0</span> &lt;0.8 Subsonic</div>' +
    '<div><span style="color:#50dc64">\\u25A0</span> 0.8-1.2 Transonic</div>' +
    '<div><span style="color:#ffa500">\\u25A0</span> 1.2-5.0 Supersonic</div>' +
    '<div><span style="color:#ff3232">\\u25A0</span> &gt;5.0 Hypersonic</div>';
  document.getElementById('cesiumContainer').appendChild(legend);
}})();

// ── Fly camera to trajectory view ──
setTimeout(() => {{
  viewer.camera.flyTo({{
    destination: Cesium.Cartesian3.fromDegrees(142, 18, 4000000),
    orientation: {{
      heading: Cesium.Math.toRadians(-10),
      pitch: Cesium.Math.toRadians(-50),
      roll: 0,
    }},
    duration: 2.0,
  }});
}}, 500);

// ── Plotly Charts ──
const darkLayout = {{
  paper_bgcolor: '#0a0e1a',
  plot_bgcolor: '#0f172a',
  font: {{ color: '#9ca3af', family: 'JetBrains Mono, monospace', size: 10 }},
  margin: {{ l: 50, r: 20, t: 30, b: 35 }},
  xaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }},
  yaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }},
  showlegend: true,
  legend: {{ font: {{ size: 9 }}, bgcolor: 'rgba(0,0,0,0)' }},
}};

// Event vertical lines
function eventShapes() {{
  return plotlyData.events.map(ev => ({{
    type: 'line', x0: ev.t, x1: ev.t, y0: 0, y1: 1, yref: 'paper',
    line: {{ color: 'rgba(255,255,100,0.3)', width: 1, dash: 'dot' }},
  }}));
}}

// Main panel (right): Altitude + Velocity
(function() {{
  const traces = [
    {{
      x: plotlyData.t, y: plotlyData.alt_km,
      name: 'Altitude [km]', yaxis: 'y',
      line: {{ color: '#60a5fa', width: 2 }},
    }},
    {{
      x: plotlyData.t, y: plotlyData.speed_kms,
      name: 'V_air [km/s]', yaxis: 'y2',
      line: {{ color: '#f59e0b', width: 2 }},
    }},
    {{
      x: plotlyData.t, y: plotlyData.v_inertial_kms,
      name: 'V_inertial [km/s]', yaxis: 'y2',
      line: {{ color: '#fb923c', width: 1, dash: 'dash' }},
    }},
  ];
  const layout = {{
    ...darkLayout,
    title: {{ text: 'Altitude & Velocity', font: {{ size: 12, color: '#60a5fa' }} }},
    xaxis: {{ ...darkLayout.xaxis, title: 'Time [s]' }},
    yaxis: {{ ...darkLayout.yaxis, title: 'Altitude [km]', side: 'left' }},
    yaxis2: {{
      title: 'Velocity [km/s]', side: 'right', overlaying: 'y',
      gridcolor: '#1e293b', showgrid: false,
    }},
    shapes: eventShapes(),
    height: null,
  }};
  Plotly.newPlot('plotly-main', traces, layout, {{ responsive: true }});
}})();

// Bottom panel: 4 subplots
(function() {{
  const el = document.getElementById('plotly-bottom');

  // Create 4 divs
  const ids = ['plot-mach', 'plot-qdyn', 'plot-temp', 'plot-mass'];
  el.innerHTML = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;">' +
    ids.map(id => `<div id="${{id}}"></div>`).join('') + '</div>';

  const subLayout = (title) => ({{
    ...darkLayout,
    title: {{ text: title, font: {{ size: 11, color: '#9ca3af' }} }},
    margin: {{ l: 45, r: 10, t: 28, b: 30 }},
    height: 200,
    showlegend: false,
    shapes: eventShapes(),
  }});

  // Mach
  Plotly.newPlot('plot-mach', [
    {{ x: plotlyData.t, y: plotlyData.mach, line: {{ color: '#a78bfa', width: 1.5 }} }},
  ], {{ ...subLayout('Mach Number'), yaxis: {{ ...darkLayout.yaxis, title: 'Mach' }} }},
  {{ responsive: true }});

  // Dynamic Pressure
  Plotly.newPlot('plot-qdyn', [
    {{ x: plotlyData.t, y: plotlyData.q_kpa, line: {{ color: '#f87171', width: 1.5 }},
      fill: 'tozeroy', fillcolor: 'rgba(248,113,113,0.1)' }},
  ], {{ ...subLayout('Dynamic Pressure'), yaxis: {{ ...darkLayout.yaxis, title: 'q [kPa]' }} }},
  {{ responsive: true }});

  // Temperature
  Plotly.newPlot('plot-temp', [
    {{ x: plotlyData.t, y: plotlyData.T_nose, line: {{ color: '#fb923c', width: 1.5 }} }},
  ], {{ ...subLayout('Nose Temperature'), yaxis: {{ ...darkLayout.yaxis, title: 'T [K]' }} }},
  {{ responsive: true }});

  // Mass + Accel
  Plotly.newPlot('plot-mass', [
    {{ x: plotlyData.t, y: plotlyData.mass_t, name: 'Mass', line: {{ color: '#34d399', width: 1.5 }} }},
    {{ x: plotlyData.t, y: plotlyData.accel_g, name: 'Accel', yaxis: 'y2',
       line: {{ color: '#e879f9', width: 1, dash: 'dash' }} }},
  ], {{
    ...subLayout('Mass & Acceleration'),
    yaxis: {{ ...darkLayout.yaxis, title: 'Mass [t]' }},
    yaxis2: {{ title: 'G', side: 'right', overlaying: 'y', showgrid: false }},
    showlegend: true,
    legend: {{ font: {{ size: 8 }}, x: 0.6, y: 0.95, bgcolor: 'rgba(0,0,0,0)' }},
  }}, {{ responsive: true }});
}})();
</script>
</body>
</html>"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run simulation first")
    parser.add_argument("--mc", type=int, default=0, help="Run MC with N samples")
    parser.add_argument("--config", default="H3-22S")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    mc_telemetry = None
    mc_stats = None

    if args.run or args.mc > 0:
        from src.vt.sixdof import H3SixDOF, H3MonteCarlo

        if args.mc > 0:
            mc = H3MonteCarlo(config=args.config)
            mc_result = mc.run(n_runs=args.mc, dt=0.1)
            telemetry = mc_result["nominal"]["telemetry"]
            events = mc_result["nominal"]["events"]
            mc_telemetry = mc_result["all_telemetry"]
            mc_stats = mc_result["statistics"]
        else:
            sim = H3SixDOF(config=args.config)
            result = sim.run(t_end=1000, dt=0.1)
            telemetry = result["telemetry"]
            events = result["events"]
    else:
        # Load from saved npz
        npz_path = "results/sixdof_telemetry.npz"
        if not os.path.exists(npz_path):
            print(f"No saved telemetry at {npz_path}. Use --run to generate.")
            sys.exit(1)
        data = dict(np.load(npz_path, allow_pickle=True))
        telemetry = data
        events = []  # No events saved in npz — extract from telemetry
        print(f"Loaded {len(telemetry['t'])} samples from {npz_path}")

    output = args.output or f"results/h3_dashboard_6dof_{args.config}.html"
    html = generate_dashboard_html(telemetry, events, mc_telemetry, mc_stats,
                                   config=args.config)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    print(f"\nDashboard: {os.path.abspath(output)}")
    print(f"  Size: {len(html)/1024:.0f} KB")


if __name__ == "__main__":
    main()
