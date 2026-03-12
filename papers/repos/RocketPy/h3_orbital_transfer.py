"""
H3 Rocket Orbital Transfer Simulation
======================================
Step E: GTO (Geostationary Transfer Orbit) + Lunar Transfer

Uses poliastro for orbital mechanics:
  1. LEO parking orbit (H3-22S insertion orbit ~300km)
  2. Hohmann transfer to GEO (35,786 km) — H3-24L GTO mission
  3. Lunar free-return trajectory
  4. Direct lunar orbit insertion

Also includes custom gravity-turn results from Step A as starting point.
"""
import math
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from astropy.time import Time

from poliastro.bodies import Earth, Moon, Sun
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.util import norm as poliastro_norm

print("=" * 70)
print("  H3 Rocket — Orbital Transfer Simulation")
print("  LEO → GTO → GEO / Lunar Transfer")
print("=" * 70)

# ============================================================
# 1. H3-22S LEO Parking Orbit (from Step A optimization)
# ============================================================
print("\n=== 1. LEO Parking Orbit (H3-22S) ===")

# From our optimized simulation: 293 x 365 km orbit
leo = Orbit.from_classical(
    Earth,
    a=(6371 + 329) * u.km,    # semi-major axis (mean of 293+365)/2 + R_earth
    ecc=0.0054 * u.one,        # from optimization
    inc=30.4 * u.deg,          # Tanegashima latitude
    raan=0 * u.deg,
    argp=0 * u.deg,
    nu=0 * u.deg,
    epoch=Time("2026-04-01 01:00:00", scale="utc"),
)

print(f"  Semi-major axis: {leo.a.to(u.km):.1f}")
print(f"  Eccentricity:    {leo.ecc:.4f}")
print(f"  Perigee:         {(leo.a * (1 - leo.ecc) - Earth.R).to(u.km):.1f}")
print(f"  Apogee:          {(leo.a * (1 + leo.ecc) - Earth.R).to(u.km):.1f}")
print(f"  Period:          {leo.period.to(u.min):.1f}")
print(f"  Inclination:     {leo.inc.to(u.deg):.1f}")

# ============================================================
# 2. Hohmann Transfer: LEO → GEO
# ============================================================
print("\n=== 2. Hohmann Transfer: LEO → GEO (H3-24L Mission) ===")

r_geo = 42164 * u.km  # GEO radius

# First circularize LEO
leo_circ = Orbit.circular(Earth, alt=300 * u.km, inc=30.4 * u.deg,
                          epoch=Time("2026-04-01 01:00:00", scale="utc"))

man_hohmann = Maneuver.hohmann(leo_circ, r_geo)
(t1, dv1), (t2, dv2) = man_hohmann.impulses

dv1_mag = poliastro_norm(dv1).to(u.km / u.s)
dv2_mag = poliastro_norm(dv2).to(u.km / u.s)
total_dv = man_hohmann.get_total_cost().to(u.km / u.s)

print(f"  LEO altitude:    300 km (circular)")
print(f"  GEO altitude:    35,786 km")
print(f"  Delta-v1 (burn): {dv1_mag:.3f} (LEO → GTO)")
print(f"  Delta-v2 (circ): {dv2_mag:.3f} (GTO → GEO)")
print(f"  Total delta-v:   {total_dv:.3f}")
print(f"  Transfer time:   {t2.to(u.hour):.1f}")

# GTO orbit (intermediate)
gto = leo_circ.apply_maneuver(man_hohmann, intermediate=True)
print(f"\n  GTO orbit:")
print(f"    Perigee: {(gto[0].a * (1 - gto[0].ecc) - Earth.R).to(u.km):.1f}")
print(f"    Apogee:  {(gto[0].a * (1 + gto[0].ecc) - Earth.R).to(u.km):.1f}")
print(f"    Period:  {gto[0].period.to(u.hour):.1f}")

# GEO orbit (final)
geo = gto[1]
print(f"\n  GEO orbit:")
print(f"    Altitude: {(geo.a - Earth.R).to(u.km):.1f}")
print(f"    Period:   {geo.period.to(u.hour):.1f}")

# ============================================================
# 3. H3 Fuel Budget for GTO
# ============================================================
print("\n=== 3. H3-24L Fuel Budget ===")

# H3-24L: 2nd stage LE-5B-3
s2_isp = 448.0  # seconds
s2_prop = 28000  # kg
s2_dry = 4000    # kg
payload_geo_transfer = 6500  # kg (H3-24L to GTO)

m_initial = s2_dry + s2_prop + payload_geo_transfer
dv_available = s2_isp * 9.81 * math.log(m_initial / (m_initial - s2_prop))

print(f"  2nd stage mass: {m_initial} kg (dry {s2_dry} + fuel {s2_prop} + payload {payload_geo_transfer})")
print(f"  Available Δv:   {dv_available:.0f} m/s")
print(f"  Required Δv (LEO→GTO): {dv1_mag.to(u.m/u.s).value:.0f} m/s")
print(f"  Margin:         {dv_available - dv1_mag.to(u.m/u.s).value:.0f} m/s")

sufficient = dv_available > dv1_mag.to(u.m / u.s).value
print(f"  Feasible:       {'YES' if sufficient else 'NO'}")

# ============================================================
# 4. Lunar Transfer (TLI — Trans-Lunar Injection)
# ============================================================
print("\n=== 4. Lunar Transfer ===")

# TLI from LEO
r_moon = 384400 * u.km  # Moon distance

# Hohmann-like to Moon distance
man_lunar = Maneuver.hohmann(leo_circ, r_moon)
(t1_l, dv1_l), (t2_l, dv2_l) = man_lunar.impulses

dv1_l_mag = poliastro_norm(dv1_l).to(u.km / u.s)
dv2_l_mag = poliastro_norm(dv2_l).to(u.km / u.s)

print(f"  TLI delta-v:     {dv1_l_mag:.3f} (from LEO)")
print(f"  LOI delta-v:     {dv2_l_mag:.3f} (lunar orbit insertion)")
print(f"  Total delta-v:   {(dv1_l_mag + dv2_l_mag):.3f}")
print(f"  Transfer time:   {t2_l.to(u.day):.1f}")

# Lunar orbit
tli_orbit = leo_circ.apply_maneuver(man_lunar, intermediate=True)
print(f"\n  Transfer orbit:")
print(f"    Perigee: {(tli_orbit[0].a * (1 - tli_orbit[0].ecc) - Earth.R).to(u.km):.1f}")
print(f"    Apogee:  {(tli_orbit[0].a * (1 + tli_orbit[0].ecc) - Earth.R).to(u.km):.1f}")
print(f"    Period:  {tli_orbit[0].period.to(u.day):.1f}")

# Free-return trajectory (simplified)
print(f"\n  Free-return trajectory:")
v_circ_leo = math.sqrt(3.986e14 / (6671e3))  # m/s
dv_tli = dv1_l_mag.to(u.m / u.s).value
v_tli = v_circ_leo + dv_tli
print(f"    V at LEO:     {v_circ_leo:.0f} m/s")
print(f"    V after TLI:  {v_tli:.0f} m/s")
print(f"    Escape vel:   {math.sqrt(2) * v_circ_leo:.0f} m/s")
print(f"    V/V_esc:      {v_tli / (math.sqrt(2) * v_circ_leo):.3f}")

# ============================================================
# 5. Bielliptic Transfer Comparison
# ============================================================
print("\n=== 5. Bielliptic vs Hohmann (for high orbits) ===")

r_b = 100000 * u.km  # intermediate apogee
man_bielliptic = Maneuver.bielliptic(leo_circ, r_b, r_geo)
dv_bielliptic = man_bielliptic.get_total_cost().to(u.km / u.s)

print(f"  Hohmann Δv to GEO:     {total_dv:.3f}")
print(f"  Bielliptic Δv to GEO:  {dv_bielliptic:.3f}")
print(f"  Hohmann is {'better' if total_dv < dv_bielliptic else 'worse'} "
      f"(ratio {dv_bielliptic/total_dv:.3f})")

# ============================================================
# Plots — 6-panel figure
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Helper: orbit to x,y arrays
def orbit_xy(orb, n=200, fraction=1.0):
    """Get x,y coordinates of orbit in perifocal frame."""
    a = orb.a.to(u.km).value
    e = orb.ecc.value
    nus = np.linspace(0, 2 * np.pi * fraction, n)
    rs = a * (1 - e**2) / (1 + e * np.cos(nus))
    xs = rs * np.cos(nus)
    ys = rs * np.sin(nus)
    return xs, ys

# (a) LEO orbit
ax = axes[0, 0]
theta = np.linspace(0, 2*np.pi, 200)
# Earth
ax.fill(6371*np.cos(theta), 6371*np.sin(theta), color='lightblue', alpha=0.5)
ax.plot(6371*np.cos(theta), 6371*np.sin(theta), 'b-', lw=1)
# LEO
xs, ys = orbit_xy(leo)
ax.plot(xs, ys, 'g-', lw=2, label=f'LEO ({leo.a.to(u.km).value - 6371:.0f} km)')
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('(a) H3-22S LEO Parking Orbit')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# (b) LEO → GTO → GEO
ax = axes[0, 1]
ax.fill(6371*np.cos(theta), 6371*np.sin(theta), color='lightblue', alpha=0.3)
ax.plot(6371*np.cos(theta), 6371*np.sin(theta), 'b-', lw=0.5)
# LEO
r_leo = (6371 + 300)
ax.plot(r_leo*np.cos(theta), r_leo*np.sin(theta), 'g-', lw=1.5, label='LEO 300km')
# GTO
xs, ys = orbit_xy(gto[0])
ax.plot(xs, ys, 'orange', lw=2, label='GTO')
# GEO
r_g = 42164
ax.plot(r_g*np.cos(theta), r_g*np.sin(theta), 'r--', lw=1.5, label='GEO')
ax.plot(r_leo, 0, 'go', ms=8)  # burn 1
ax.plot(-r_g, 0, 'r*', ms=12)   # burn 2 (approximate)
ax.annotate(f'TLI burn\nΔv={dv1_mag.value:.2f} km/s', (r_leo+500, 1500), fontsize=7)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_title('(b) Hohmann: LEO → GTO → GEO')
ax.legend(fontsize=8, loc='upper left')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# (c) Lunar transfer
ax = axes[0, 2]
# Earth (small dot at scale)
ax.plot(0, 0, 'bo', ms=8, label='Earth')
# LEO (tiny ring)
ax.plot(6671*np.cos(theta)/1000, 6671*np.sin(theta)/1000, 'g-', lw=0.5)
# Transfer ellipse
xs, ys = orbit_xy(tli_orbit[0], fraction=0.5)
ax.plot(xs/1000, ys/1000, 'orange', lw=2.5, label='Transfer orbit')
# Moon orbit
r_m = 384.4  # thousand km
ax.plot(r_m*np.cos(theta), r_m*np.sin(theta), 'gray', lw=0.5, ls='--', alpha=0.5)
ax.plot(r_m, 0, 'ko', ms=10, label='Moon')
ax.annotate('Moon', (r_m + 10, 10), fontsize=9)
ax.annotate(f'TLI Δv={dv1_l_mag.value:.2f} km/s', (10, -50), fontsize=8)
ax.set_xlabel("x (×10³ km)")
ax.set_ylabel("y (×10³ km)")
ax.set_title('(c) Trans-Lunar Injection')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# (d) Delta-v budget comparison
ax = axes[1, 0]
missions = ['LEO→GTO', 'GTO→GEO', 'LEO→GEO\n(total)', 'TLI\n(LEO→Moon)', 'LOI\n(Moon capture)']
dvs = [dv1_mag.value, dv2_mag.value, total_dv.value,
       dv1_l_mag.value, dv2_l_mag.value]
colors = ['steelblue', 'steelblue', 'navy', 'firebrick', 'firebrick']
bars = ax.bar(missions, dvs, color=colors, alpha=0.8)
for bar, dv in zip(bars, dvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{dv:.2f}', ha='center', fontsize=8)
ax.set_ylabel('Delta-v (km/s)')
ax.set_title('(d) Delta-v Budget')
ax.grid(True, alpha=0.3, axis='y')

# (e) H3 payload capability vs delta-v
ax = axes[1, 1]
dvs_range = np.linspace(0, 5000, 100)  # m/s
isp = 448
m_stage = s2_dry + s2_prop
payloads = []
for dv in dvs_range:
    # m_payload = m_total * exp(-dv/(isp*g)) - m_dry_stage
    # m_total = m_stage + m_payload
    # m_payload = (m_stage + m_payload) * exp(-dv/(Isp*g)) - s2_dry
    # m_payload * (1 - exp(-dv/(Isp*g))) = m_stage * exp(-dv/(Isp*g)) - s2_dry
    exp_term = math.exp(-dv / (isp * 9.81))
    if (1 - exp_term) > 0:
        m_pl = (m_stage * exp_term - s2_dry) / (1 - exp_term)
        payloads.append(max(0, m_pl))
    else:
        payloads.append(0)

ax.plot(dvs_range / 1000, [p/1000 for p in payloads], 'b-', lw=2)
# Mark key missions
ax.axvline(x=dv1_mag.value, color='orange', ls='--', alpha=0.7)
ax.text(dv1_mag.value + 0.05, 8, 'GTO', fontsize=8, color='orange')
ax.axvline(x=dv1_l_mag.value, color='red', ls='--', alpha=0.7)
ax.text(dv1_l_mag.value + 0.05, 8, 'TLI', fontsize=8, color='red')
ax.set_xlabel('Delta-v (km/s)')
ax.set_ylabel('Payload Mass (ton)')
ax.set_title('(e) H3 2nd Stage Payload vs Delta-v')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 10)

# (f) Transfer time comparison
ax = axes[1, 2]
transfers = {
    'LEO→GTO\n(Hohmann)': t2.to(u.hour).value,
    'LEO→GEO\n(full)': t2.to(u.hour).value,
    'LEO→Moon\n(Hohmann)': t2_l.to(u.hour).value,
}
bars = ax.bar(transfers.keys(), transfers.values(), color=['steelblue', 'navy', 'firebrick'], alpha=0.8)
for bar, val in zip(bars, transfers.values()):
    if val > 24:
        label = f'{val/24:.1f} days'
    else:
        label = f'{val:.1f} hours'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            label, ha='center', fontsize=8)
ax.set_ylabel('Transfer Time (hours)')
ax.set_title('(f) Transfer Duration')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f"H3 Rocket — Orbital Transfer Analysis (poliastro)\n"
    f"LEO 300km → GTO (Δv={dv1_mag.value:.2f} km/s) → GEO (total Δv={total_dv.value:.2f} km/s)  |  "
    f"TLI Δv={dv1_l_mag.value:.2f} km/s ({t2_l.to(u.day).value:.1f} days)",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig("h3_orbital_transfer.png", dpi=150, bbox_inches="tight")
print(f"\nFigure saved: h3_orbital_transfer.png")

# Summary table
print(f"\n{'='*60}")
print(f"  Mission Summary")
print(f"{'='*60}")
print(f"  {'Mission':<25} {'Δv (km/s)':<12} {'Time':<15} {'Payload':<10}")
print(f"  {'-'*55}")
print(f"  {'H3-22S → LEO 300km':<25} {'(launch)':<12} {'~5 min':<15} {'~4 ton':<10}")
print(f"  {'LEO → GTO':<25} {dv1_mag.value:<12.3f} {t2.to(u.hour).value/2:<12.1f}{'h':<3} {'6.5 ton':<10}")
print(f"  {'GTO → GEO':<25} {dv2_mag.value:<12.3f} {t2.to(u.hour).value/2:<12.1f}{'h':<3} {'(sat)':<10}")
print(f"  {'LEO → Moon (TLI)':<25} {dv1_l_mag.value:<12.3f} {t2_l.to(u.day).value:<12.1f}{'d':<3} {'~1 ton':<10}")
print(f"  {'Moon orbit (LOI)':<25} {dv2_l_mag.value:<12.3f} {'':<15} {'(sat)':<10}")
