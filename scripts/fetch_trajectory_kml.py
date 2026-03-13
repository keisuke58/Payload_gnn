#!/usr/bin/env python3
"""
JPL Horizons API から探査機の軌道データを取得し、Google Earth KML に変換する。
地上投影 (ground track) として表示。

対象ミッション:
  1. Artemis I (Orion) — 地球→月 往復  (ID: -1023)
  2. JAXA SLIM — 地球→月             (ID: -240)
  3. Mars 2020 (Perseverance) — 地球→火星 (ID: -168)
"""

import urllib.request
import urllib.parse
import math
import re
import os

# ── 定数 ──
RE = 6371.0  # 地球半径 [km]

# ── ミッション定義 ──
MISSIONS = [
    {
        "name": "Artemis I (Orion)",
        "command": "-1023",
        "start": "2022-11-17",
        "stop": "2022-12-11 17:00",
        "step": "1 h",
        "center": "500@399",
        "description": "NASA Artemis I — Orion MPCV lunar flyby mission\n"
                       "Launch: 2022-11-16 from KSC LC-39B\n"
                       "Splashdown: 2022-12-11\n"
                       "Max distance: ~432,000 km (beyond Moon orbit)",
        "color": "ff0000ff",   # red (AABBGGRR)
        "width": 3,
    },
    {
        "name": "JAXA SLIM",
        "command": "-240",
        "start": "2023-09-07 01:00",
        "stop": "2024-01-20 00:00",
        "step": "3 h",
        "center": "500@399",
        "description": "JAXA Smart Lander for Investigating Moon\n"
                       "Launch: 2023-09-07 from Tanegashima\n"
                       "Moon landing: 2024-01-19\n"
                       "Max distance: ~1,380,000 km",
        "color": "ff00aaff",   # orange
        "width": 3,
    },
    {
        "name": "Mars 2020 (Perseverance)",
        "command": "-168",
        "start": "2020-07-30 13:00",
        "stop": "2021-02-18 20:00",
        "step": "12 h",
        "center": "500@399",
        "description": "NASA Mars 2020 — Perseverance rover + Ingenuity helicopter\n"
                       "Launch: 2020-07-30 from Cape Canaveral\n"
                       "Mars landing: 2021-02-18 Jezero Crater\n"
                       "Max distance: ~204,000,000 km",
        "color": "ff3333ff",   # red-ish
        "width": 3,
    },
]


def fetch_horizons(command, start, stop, step, center="500@399"):
    """JPL Horizons API からベクトルエフェメリスを取得"""
    params = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "CSV_FORMAT": "'YES'",
        "VEC_TABLE": "'2'",
        "REF_PLANE": "'FRAME'",
        "VEC_LABELS": "'YES'",
    }
    parts = []
    for k, v in params.items():
        v_enc = v.replace(" ", "%20")
        parts.append(f"{k}={v_enc}")
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + "&".join(parts)
    print(f"  Fetching: {command} ...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8")
    return text


def parse_horizons_vectors(text):
    """$$SOE〜$$EOE の CSV データをパース → [(jd, cal_date, x, y, z), ...]"""
    m = re.search(r'\$\$SOE\s*\n(.*?)\$\$EOE', text, re.DOTALL)
    if not m:
        raise ValueError("Could not find $$SOE/$$EOE in response")
    data = []
    for line in m.group(1).strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue
        try:
            jd = float(parts[0])
            cal_date = parts[1].strip()
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            data.append((jd, cal_date, x, y, z))
        except (ValueError, IndexError):
            continue
    return data


def jd_to_gmst_deg(jd):
    """Julian Date → GMST [degrees]"""
    T = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
           0.000387933 * T**2 - T**3 / 38710000.0
    return gmst % 360.0


def eci_to_ground_track(data):
    """
    ECI (J2000 equatorial) → ground track (lon, lat) on Earth surface.
    地球自転を考慮して sub-spacecraft point を計算。
    """
    coords = []
    for jd, cal, x, y, z in data:
        r = math.sqrt(x**2 + y**2 + z**2)
        if r < 1.0:
            continue
        # ECI → ECEF (地球自転補正)
        gmst = math.radians(jd_to_gmst_deg(jd))
        x_ecef = x * math.cos(gmst) + y * math.sin(gmst)
        y_ecef = -x * math.sin(gmst) + y * math.cos(gmst)
        z_ecef = z
        # ECEF → geodetic (球体近似)
        lat = math.degrees(math.asin(z_ecef / r))
        lon = math.degrees(math.atan2(y_ecef, x_ecef))
        dist_km = r
        coords.append((lon, lat, dist_km, cal))
    return coords


def split_at_dateline(coords):
    """
    経度が ±180° を跨ぐ箇所でラインを分割。
    Google Earth で線が地球を横断しないように。
    """
    segments = []
    current = []
    for i, (lon, lat, dist, cal) in enumerate(coords):
        if current and abs(lon - current[-1][0]) > 170:
            segments.append(current)
            current = []
        current.append((lon, lat, dist, cal))
    if current:
        segments.append(current)
    return segments


def make_kml(missions_data):
    """複数ミッションの ground track を1つの KML にまとめる"""
    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"'
        ' xmlns:gx="http://www.google.com/kml/ext/2.2">',
        '<Document>',
        '  <name>Moon &amp; Mars Mission Ground Tracks</name>',
        '  <description>Sub-spacecraft ground tracks from JPL Horizons data.\n'
        '  Shows the point on Earth directly below each spacecraft.</description>',
        '  <open>1</open>',
    ]

    for mission, coords in missions_data:
        name = mission["name"]
        color = mission["color"]
        width = mission.get("width", 3)
        desc = mission["description"]
        sid = f"s_{name.replace(' ', '_').replace('(', '').replace(')', '')}"

        # Styles
        kml.append(f'  <Style id="{sid}">')
        kml.append(f'    <LineStyle><color>{color}</color><width>{width}</width></LineStyle>')
        kml.append(f'  </Style>')
        kml.append(f'  <Style id="{sid}_pt">')
        kml.append(f'    <IconStyle><color>{color}</color><scale>0.8</scale></IconStyle>')
        kml.append(f'  </Style>')

        kml.append(f'  <Folder>')
        kml.append(f'    <name>{name}</name>')
        kml.append(f'    <description><![CDATA[{desc}]]></description>')
        kml.append(f'    <open>1</open>')

        # Split at dateline to avoid wraparound lines
        segments = split_at_dateline(coords)

        for si, seg in enumerate(segments):
            suffix = f" (seg {si+1})" if len(segments) > 1 else ""
            kml.append(f'    <Placemark>')
            kml.append(f'      <name>{name} Ground Track{suffix}</name>')
            kml.append(f'      <styleUrl>#{sid}</styleUrl>')
            kml.append(f'      <LineString>')
            kml.append(f'        <tessellate>1</tessellate>')
            kml.append(f'        <altitudeMode>clampToGround</altitudeMode>')
            kml.append(f'        <coordinates>')
            for lon, lat, dist, cal in seg:
                kml.append(f'          {lon:.4f},{lat:.4f},0')
            kml.append(f'        </coordinates>')
            kml.append(f'      </LineString>')
            kml.append(f'    </Placemark>')

        # Key waypoints: start, furthest, end, plus periodic markers
        if coords:
            # Launch
            lon0, lat0, d0, c0 = coords[0]
            kml.append(f'    <Placemark>')
            kml.append(f'      <name>Launch — {c0.strip()}</name>')
            kml.append(f'      <description>Distance: {d0:,.0f} km</description>')
            kml.append(f'      <styleUrl>#{sid}_pt</styleUrl>')
            kml.append(f'      <Point><coordinates>{lon0:.4f},{lat0:.4f},0</coordinates></Point>')
            kml.append(f'    </Placemark>')

            # Furthest point
            max_i = max(range(len(coords)), key=lambda i: coords[i][2])
            lonm, latm, dm, cm = coords[max_i]
            kml.append(f'    <Placemark>')
            kml.append(f'      <name>Furthest — {dm:,.0f} km — {cm.strip()}</name>')
            kml.append(f'      <description>Maximum distance from Earth center</description>')
            kml.append(f'      <styleUrl>#{sid}_pt</styleUrl>')
            kml.append(f'      <Point><coordinates>{lonm:.4f},{latm:.4f},0</coordinates></Point>')
            kml.append(f'    </Placemark>')

            # End
            lonE, latE, dE, cE = coords[-1]
            kml.append(f'    <Placemark>')
            kml.append(f'      <name>End — {cE.strip()}</name>')
            kml.append(f'      <description>Distance: {dE:,.0f} km</description>')
            kml.append(f'      <styleUrl>#{sid}_pt</styleUrl>')
            kml.append(f'      <Point><coordinates>{lonE:.4f},{latE:.4f},0</coordinates></Point>')
            kml.append(f'    </Placemark>')

            # Weekly markers (every ~7 days worth of points)
            step_hours = {"1 h": 1, "3 h": 3, "6 h": 6, "12 h": 12, "30 min": 0.5}
            h = step_hours.get(mission["step"], 6)
            pts_per_week = int(7 * 24 / h)
            if pts_per_week > 0:
                for wi in range(pts_per_week, len(coords), pts_per_week):
                    lon_w, lat_w, d_w, c_w = coords[wi]
                    kml.append(f'    <Placemark>')
                    kml.append(f'      <name>{c_w.strip()} — {d_w:,.0f} km</name>')
                    kml.append(f'      <styleUrl>#{sid}_pt</styleUrl>')
                    kml.append(f'      <Point><coordinates>{lon_w:.4f},{lat_w:.4f},0</coordinates></Point>')
                    kml.append(f'    </Placemark>')

        kml.append(f'  </Folder>')

    kml.append('</Document>')
    kml.append('</kml>')
    return '\n'.join(kml)


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "papers", "repos", "RocketPy"
    )

    missions_data = []

    for mission in MISSIONS:
        print(f"\n=== {mission['name']} ===")
        try:
            text = fetch_horizons(
                mission["command"], mission["start"],
                mission["stop"], mission["step"], mission["center"],
            )
            data = parse_horizons_vectors(text)
            print(f"  Parsed {len(data)} data points")
            if not data:
                print(f"  WARNING: No data parsed. Snippet:\n{text[:500]}")
                continue

            dists = [math.sqrt(x**2 + y**2 + z**2) for _, _, x, y, z in data]
            print(f"  Distance range: {min(dists):,.0f} — {max(dists):,.0f} km")

            coords = eci_to_ground_track(data)
            print(f"  Ground track points: {len(coords)}")
            missions_data.append((mission, coords))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not missions_data:
        print("\nNo data fetched.")
        return

    # Combined KML
    kml = make_kml(missions_data)
    out = os.path.join(output_dir, "moon_mars_trajectories.kml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(kml)
    print(f"\n=> {out}")

    # Individual KMLs
    for mission, coords in missions_data:
        safe = mission["name"].replace(" ", "_").replace("(", "").replace(")", "")
        individual = make_kml([(mission, coords)])
        p = os.path.join(output_dir, f"trajectory_{safe}.kml")
        with open(p, "w", encoding="utf-8") as f:
            f.write(individual)
        print(f"   {p}")

    print(f"\nDone. {len(missions_data)} missions.")


if __name__ == "__main__":
    main()
