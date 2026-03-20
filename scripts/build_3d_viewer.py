#!/usr/bin/env python3
"""Build fairing 3D viewer HTML with embedded CSV data.

Usage: python3 scripts/build_3d_viewer.py

Reads separation snapshot CSVs, filters SKIN nodes, downsamples to ~5000,
and generates a self-contained Three.js HTML viewer.
"""
import csv
import json
import os

def load_skin_downsampled(filepath, target=5000):
    """Load CSV, filter SKIN instances, downsample to target count."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row['instance']
            if 'INNERSKIN' in inst or 'OUTERSKIN' in inst:
                rows.append([
                    round(float(row['u_x_mm']), 4),
                    round(float(row['u_y_mm']), 4),
                    round(float(row['u_z_mm']), 4),
                    round(float(row['u_mag_mm']), 4),
                ])
    total = len(rows)
    step = max(1, total // target)
    sampled = rows[::step]
    print(f"  {os.path.basename(filepath)}: {total} SKIN nodes, step={step}, sampled={len(sampled)}")
    return sampled


def build_html(normal, stuck6, n_stats, s_stats):
    """Generate the complete HTML string."""
    normal_json = json.dumps(normal)
    stuck6_json = json.dumps(stuck6)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>H3 Fairing Separation - 3D Displacement Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0e1a; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; }}
  #canvas-container {{ width: 100vw; height: 100vh; position: relative; }}
  canvas {{ display: block; }}

  #overlay {{
    position: absolute; top: 0; left: 0; width: 100%; padding: 20px 30px;
    pointer-events: none; z-index: 10;
  }}
  #overlay h1 {{
    font-size: 22px; font-weight: 700; color: #fff;
    text-shadow: 0 2px 8px rgba(0,0,0,0.7);
    margin-bottom: 4px;
  }}
  #overlay .subtitle {{
    font-size: 13px; color: #8899bb; letter-spacing: 0.5px;
  }}

  #kpi-panel {{
    position: absolute; top: 80px; left: 30px;
    background: rgba(10,14,26,0.85); border: 1px solid #1a2540;
    border-radius: 10px; padding: 16px 20px; min-width: 220px;
    backdrop-filter: blur(8px); z-index: 10;
  }}
  #kpi-panel .kpi-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 8px;
  }}
  #kpi-panel .kpi-label {{ font-size: 12px; color: #6688aa; }}
  #kpi-panel .kpi-value {{ font-size: 16px; font-weight: 600; color: #4fc3f7; }}
  #kpi-panel .kpi-value.warn {{ color: #ff8a65; }}

  #controls {{
    position: absolute; top: 80px; right: 30px; z-index: 10;
  }}
  #case-select {{
    background: rgba(10,14,26,0.9); color: #e0e0e0; border: 1px solid #2a3a5a;
    border-radius: 6px; padding: 8px 14px; font-size: 14px; cursor: pointer;
    outline: none;
  }}
  #case-select:hover {{ border-color: #4fc3f7; }}
  #case-select option {{ background: #0a0e1a; }}

  #colorbar {{
    position: absolute; bottom: 40px; left: 50%; transform: translateX(-50%);
    z-index: 10; text-align: center;
  }}
  #colorbar .bar {{
    width: 300px; height: 16px; border-radius: 3px;
    background: linear-gradient(to right,
      #000033, #0000ff, #00ccff, #00ff66, #ffff00, #ff6600, #ff0000);
    border: 1px solid #2a3a5a;
  }}
  #colorbar .labels {{
    display: flex; justify-content: space-between; width: 300px;
    margin-top: 4px; font-size: 11px; color: #8899bb;
  }}
  #colorbar .title {{
    font-size: 12px; color: #aabbcc; margin-bottom: 4px; letter-spacing: 0.5px;
  }}

  #loading {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    font-size: 16px; color: #4fc3f7; z-index: 20;
  }}

  #help-text {{
    position: absolute; bottom: 80px; right: 30px; z-index: 10;
    font-size: 11px; color: #556688; text-align: right; line-height: 1.6;
  }}
</style>
</head>
<body>

<div id="canvas-container">
  <div id="loading">Loading 3D viewer...</div>

  <div id="overlay">
    <h1>H3 Fairing Separation &mdash; Displacement Field</h1>
    <div class="subtitle">CFRP/Al-Honeycomb Fairing &middot; t = 0.200 s &middot; Inner + Outer Skin</div>
  </div>

  <div id="kpi-panel">
    <div class="kpi-row">
      <span class="kpi-label">Nodes</span>
      <span class="kpi-value" id="kpi-nodes">--</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-label">Max |u| (mm)</span>
      <span class="kpi-value" id="kpi-max">--</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-label">Mean |u| (mm)</span>
      <span class="kpi-value" id="kpi-mean">--</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-label">Case</span>
      <span class="kpi-value" id="kpi-case">--</span>
    </div>
  </div>

  <div id="controls">
    <select id="case-select">
      <option value="normal">Normal Separation</option>
      <option value="stuck6">Stuck-6 (Anomaly)</option>
    </select>
  </div>

  <div id="colorbar">
    <div class="title">Displacement Magnitude |u| (mm)</div>
    <div class="bar"></div>
    <div class="labels">
      <span id="cb-min">0.0</span>
      <span id="cb-mid">--</span>
      <span id="cb-max">--</span>
    </div>
  </div>

  <div id="help-text">
    Left-drag: Orbit<br>
    Right-drag: Pan<br>
    Scroll: Zoom
  </div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

// ============================================================
// Embedded data: [u_x, u_y, u_z, u_mag] per node (mm)
// ============================================================
const DATA_NORMAL = {normal_json};
const DATA_STUCK6 = {stuck6_json};

const DATASETS = {{
  normal: {{ data: DATA_NORMAL, label: "Normal", maxU: {n_stats['max']}, meanU: {n_stats['mean']} }},
  stuck6: {{ data: DATA_STUCK6, label: "Stuck-6", maxU: {s_stats['max']}, meanU: {s_stats['mean']} }}
}};

// ============================================================
// Hot colormap
// ============================================================
function hotColormap(t) {{
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.2) {{
    const s = t / 0.2;
    r = 0; g = 0; b = 0.2 + 0.8 * s;
  }} else if (t < 0.4) {{
    const s = (t - 0.2) / 0.2;
    r = 0; g = s * 0.8; b = 1.0 - 0.2 * s;
  }} else if (t < 0.6) {{
    const s = (t - 0.4) / 0.2;
    r = s * 0.2; g = 0.8 + 0.2 * s; b = 0.8 - 0.8 * s;
  }} else if (t < 0.8) {{
    const s = (t - 0.6) / 0.2;
    r = 0.2 + 0.8 * s; g = 1.0 - 0.2 * s; b = 0;
  }} else {{
    const s = (t - 0.8) / 0.2;
    r = 1.0; g = 0.8 - 0.8 * s; b = 0;
  }}
  return [r, g, b];
}}

// ============================================================
// Three.js setup
// ============================================================
let scene, camera, renderer, controls, points;
let currentCase = 'normal';

function init() {{
  const container = document.getElementById('canvas-container');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e1a);
  scene.fog = new THREE.FogExp2(0x0a0e1a, 0.0006);

  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 5000);
  camera.position.set(150, 100, 200);

  renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.6;
  controls.zoomSpeed = 1.2;
  controls.panSpeed = 0.5;
  controls.minDistance = 5;
  controls.maxDistance = 1500;

  // Grid
  const grid = new THREE.GridHelper(400, 40, 0x1a2540, 0x0f1525);
  grid.position.y = -80;
  scene.add(grid);

  // Axes labels via small colored lines
  const axLen = 20;
  const axMat = [
    new THREE.LineBasicMaterial({{ color: 0xff4444 }}),
    new THREE.LineBasicMaterial({{ color: 0x44ff44 }}),
    new THREE.LineBasicMaterial({{ color: 0x4444ff }})
  ];
  const axDirs = [
    [new THREE.Vector3(0,0,0), new THREE.Vector3(axLen,0,0)],
    [new THREE.Vector3(0,0,0), new THREE.Vector3(0,axLen,0)],
    [new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,axLen)]
  ];
  axDirs.forEach((pts, i) => {{
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const l = new THREE.Line(g, axMat[i]);
    l.position.set(-120, -80, -120);
    scene.add(l);
  }});

  buildPointCloud('normal');

  window.addEventListener('resize', onResize);
  document.getElementById('case-select').addEventListener('change', (e) => {{
    currentCase = e.target.value;
    buildPointCloud(currentCase);
  }});

  document.getElementById('loading').style.display = 'none';
  animate();
}}

function buildPointCloud(caseName) {{
  if (points) {{
    scene.remove(points);
    points.geometry.dispose();
    points.material.dispose();
  }}

  const ds = DATASETS[caseName];
  const data = ds.data;
  const n = data.length;
  const maxU = ds.maxU;
  const meanU = ds.meanU;

  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {{
    const ux = data[i][0];
    const uy = data[i][1];
    const uz = data[i][2];
    const um = data[i][3];

    // Map displacements to 3D positions (scale for visibility)
    // x -> x, z -> y (up), y -> z
    const scale = 10.0;
    positions[i * 3]     = ux * scale;
    positions[i * 3 + 1] = uz * scale;
    positions[i * 3 + 2] = uy * scale;

    const t = um / maxU;
    const [r, g, b] = hotColormap(t);
    colors[i * 3]     = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }}

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  // Center
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  geometry.translate(-center.x, -center.y, -center.z);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const material = new THREE.PointsMaterial({{
    size: 2.0,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.9,
  }});

  points = new THREE.Points(geometry, material);
  scene.add(points);

  // Update KPI
  document.getElementById('kpi-nodes').textContent = n.toLocaleString();
  document.getElementById('kpi-max').textContent = maxU.toFixed(2);
  document.getElementById('kpi-mean').textContent = meanU.toFixed(2);
  document.getElementById('kpi-case').textContent = ds.label;
  document.getElementById('kpi-case').className = caseName === 'stuck6' ? 'kpi-value warn' : 'kpi-value';

  // Colorbar
  document.getElementById('cb-min').textContent = '0.0';
  document.getElementById('cb-mid').textContent = (maxU / 2).toFixed(1);
  document.getElementById('cb-max').textContent = maxU.toFixed(1);

  // Camera
  const radius = geometry.boundingSphere.radius;
  camera.position.set(radius * 1.2, radius * 0.8, radius * 1.2);
  controls.target.set(0, 0, 0);
  controls.update();
}}

function onResize() {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

init();
</script>
</body>
</html>
'''


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'separation')

    print("Loading Normal...")
    normal = load_skin_downsampled(os.path.join(base, 'Sep-v2-Normal_snapshot_t0.200.csv'))
    print("Loading Stuck6...")
    stuck6 = load_skin_downsampled(os.path.join(base, 'Sep-v2-Stuck6_snapshot_t0.200.csv'))

    n_umag = [r[3] for r in normal]
    s_umag = [r[3] for r in stuck6]
    n_stats = {'max': round(max(n_umag), 4), 'mean': round(sum(n_umag)/len(n_umag), 4)}
    s_stats = {'max': round(max(s_umag), 4), 'mean': round(sum(s_umag)/len(s_umag), 4)}

    print(f"Normal: max={n_stats['max']}, mean={n_stats['mean']}")
    print(f"Stuck6: max={s_stats['max']}, mean={s_stats['mean']}")

    html = build_html(normal, stuck6, n_stats, s_stats)

    out_path = os.path.join(base, 'fairing_3d_viewer.html')
    with open(out_path, 'w') as f:
        f.write(html)

    fsize = os.path.getsize(out_path)
    print(f"\nWritten: {out_path}")
    print(f"Size: {fsize / 1024:.0f} KB ({fsize / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
