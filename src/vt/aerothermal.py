#!/usr/bin/env python3
"""
H3 Virtual Twin — Aerothermal Module (A5)

フェアリング / ノーズコーン表面の空力加熱 1D モデル。
飛行中の表面温度履歴を計算し、TPS 設計・SHM 連携に使用。

手法:
  - Sutton-Graves: よどみ点加熱 (convective)
  - Tauber-Sutton: 輻射加熱 (高速域)
  - Fay-Riddell 補正 (化学反応効果)
  - 半経験式による周方向分布 (θ 依存)
  - 1D 壁面熱伝導 (lumped capacitance or 1D transient)

H3 フェアリング:
  - CFRP/Al-Honeycomb サンドイッチ構造
  - 先端: アブレーション or コルク系 TPS
  - ノーズコーン長 ~8 m, 楕円形状

References:
  - Sutton & Graves, "A General Stagnation-Point Convective
    Heating Equation" (NASA TR R-376, 1971)
  - Tauber & Sutton, "Stagnation-Point Radiative Heating
    Relations" (J. Spacecraft, 1991)
  - Anderson, "Hypersonic and High-Temperature Gas Dynamics"
  - JAXA H3 Rocket User's Manual
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .aerodynamics import atmosphere_isa


# ── 物理定数 ──
STEFAN_BOLTZMANN = 5.67e-8   # [W/(m²·K⁴)]
R_AIR = 287.058              # [J/(kg·K)]


# ═══════════════════════════════════════════════════════════════
# フェアリング TPS 材料
# ═══════════════════════════════════════════════════════════════

@dataclass
class TPSMaterial:
    """TPS / フェアリング壁面の熱物性"""
    name: str = "Cork-Phenolic"
    rho: float = 520.0          # 密度 [kg/m³]
    cp: float = 1200.0          # 比熱 [J/(kg·K)]
    k: float = 0.25             # 熱伝導率 [W/(m·K)]
    emissivity: float = 0.85    # 放射率 [-]
    T_max: float = 600.0        # 使用限界温度 [K]
    thickness: float = 0.005    # TPS 厚さ [m]


# 代表的な TPS 材料
TPS_CORK = TPSMaterial("Cork-Phenolic", 520, 1200, 0.25, 0.85, 600, 0.005)
TPS_CFRP = TPSMaterial("CFRP-Skin", 1600, 900, 5.0, 0.80, 450, 0.003)
TPS_ABLATOR = TPSMaterial("SLA-561V", 260, 1000, 0.12, 0.90, 2500, 0.010)


# ═══════════════════════════════════════════════════════════════
# フェアリング形状
# ═══════════════════════════════════════════════════════════════

@dataclass
class FairingGeometry:
    """ノーズコーン / フェアリング形状"""
    nose_radius: float = 0.80       # ノーズ先端曲率半径 [m]
    nose_length: float = 8.0        # ノーズ長さ [m]
    cylinder_radius: float = 2.6    # 円筒部半径 [m]
    cylinder_length: float = 7.0    # 円筒部長さ [m]
    shape: str = "elliptical"       # "elliptical", "ogive", "conical"

    def surface_points(self, n: int = 50) -> tuple:
        """
        ノーズコーン表面の (x, r, theta_cone) を生成

        Returns:
            x[m], r[m], theta[deg] (cone half-angle at each point)
        """
        x = np.linspace(0.01, self.nose_length + self.cylinder_length, n)
        r = np.zeros(n)
        theta = np.zeros(n)

        for i, xi in enumerate(x):
            if xi <= self.nose_length:
                # ノーズ部
                t = xi / self.nose_length
                if self.shape == "elliptical":
                    r[i] = self.cylinder_radius * math.sqrt(
                        1 - (1 - t)**2)
                elif self.shape == "ogive":
                    # Tangent ogive
                    rho_og = (self.cylinder_radius**2 + self.nose_length**2
                              ) / (2 * self.cylinder_radius)
                    r[i] = math.sqrt(
                        rho_og**2 - (self.nose_length - xi)**2
                    ) - (rho_og - self.cylinder_radius)
                    r[i] = max(r[i], 0.01)
                else:  # conical
                    r[i] = self.cylinder_radius * t
                r[i] = max(r[i], 0.01)
            else:
                r[i] = self.cylinder_radius

            # Local cone half-angle
            if i > 0:
                dr = r[i] - r[i - 1]
                dx = x[i] - x[i - 1]
                theta[i] = math.degrees(math.atan2(dr, dx))
            else:
                theta[i] = 45.0  # nose tip approximation

        return x, r, theta

    @property
    def nose_tip_radius(self) -> float:
        """ノーズ先端の実効曲率半径"""
        if self.shape == "elliptical":
            # 楕円の先端曲率半径: R = b²/a
            a = self.nose_length
            b = self.cylinder_radius
            return b**2 / a
        return self.nose_radius


# ═══════════════════════════════════════════════════════════════
# 空力加熱モデル
# ═══════════════════════════════════════════════════════════════

@dataclass
class H3Aerothermal:
    """
    H3 フェアリング空力加熱 1D モデル

    Usage:
        model = H3Aerothermal()
        # 単一時刻
        result = model.heating_rate(mach=5.0, altitude=40000)
        # 飛行プロファイル
        history = model.temperature_history(t, alt, mach)
    """
    fairing: FairingGeometry = field(default_factory=FairingGeometry)
    tps_nose: TPSMaterial = field(default_factory=lambda: TPS_CORK)
    tps_body: TPSMaterial = field(default_factory=lambda: TPS_CFRP)

    # ── よどみ点加熱: Sutton-Graves ──

    def stagnation_heating(self, rho: float, V: float,
                           R_n: float = None) -> float:
        """
        Sutton-Graves よどみ点対流加熱率

        q_s = C * sqrt(rho / R_n) * V³

        Args:
            rho: 大気密度 [kg/m³]
            V:   飛行速度 [m/s]
            R_n: ノーズ曲率半径 [m] (default: fairing nose tip)

        Returns:
            q_conv [W/m²] — よどみ点対流熱流束
        """
        if R_n is None:
            R_n = self.fairing.nose_tip_radius
        R_n = max(R_n, 0.01)

        # Sutton-Graves 定数 (空気)
        C_sg = 1.7415e-4  # [kg^0.5 / m]

        q = C_sg * math.sqrt(rho / R_n) * V**3
        return q

    def radiation_heating(self, rho: float, V: float,
                          R_n: float = None) -> float:
        """
        Tauber-Sutton よどみ点輻射加熱率 (高速域)

        主に V > 6 km/s で重要 (H3 では小さい)

        Args:
            rho: 大気密度 [kg/m³]
            V:   飛行速度 [m/s]
            R_n: ノーズ曲率半径 [m]

        Returns:
            q_rad [W/m²]
        """
        if R_n is None:
            R_n = self.fairing.nose_tip_radius

        # H3 の最大速度 ~7.5 km/s → 輻射加熱はわずか
        V_km = V / 1000.0
        if V_km < 4.0:
            return 0.0

        # Tauber-Sutton (簡易)
        C_rad = 4.736e4  # [W/m²] fitting constant
        f = 1.072e6 * rho**1.22 * R_n**1.0
        q = C_rad * f * V_km**8.5 * 1e-6
        return max(q, 0.0)

    # ── 表面分布 ──

    def surface_distribution(self, q_stag: float, x: np.ndarray,
                             r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        よどみ点加熱率から表面に沿った分布を計算

        半経験式: q(s)/q_stag ≈ cos^n(θ_eff)

        Args:
            q_stag: よどみ点熱流束 [W/m²]
            x, r, theta: 表面形状 (from FairingGeometry.surface_points)

        Returns:
            q_surface [W/m²] array
        """
        q = np.zeros_like(x)
        L_nose = self.fairing.nose_length

        for i in range(len(x)):
            if x[i] <= L_nose:
                # ノーズ部: 先端から離れるほど減衰
                s_ratio = x[i] / L_nose
                # Lees distribution: q/q_stag ~ (r_stag/r)^0.5 for blunt body
                # Modified for slender nose
                if s_ratio < 0.05:
                    q[i] = q_stag
                else:
                    # 半経験的減衰
                    q[i] = q_stag * (0.1 / s_ratio)**0.5 * 0.3
                    q[i] = min(q[i], q_stag)
            else:
                # 円筒部: 境界層による減衰 (flat plate analogy)
                x_from_nose = x[i] - L_nose
                Re_x = max(1e5, 1e7 * (x[i] / 63.0))  # rough estimate
                # Turbulent: St ~ 0.03 * Re_x^(-0.2)
                St = 0.03 * Re_x**(-0.2)
                q[i] = q_stag * St * 5.0  # normalized

        return q

    # ── 単一時刻の加熱率 ──

    def heating_rate(self, mach: float, altitude: float) -> dict:
        """
        指定条件での加熱率を計算

        Args:
            mach: マッハ数
            altitude: 高度 [m]

        Returns:
            dict: q_stag_conv, q_stag_rad, q_stag_total [W/m²],
                  T_recovery[K], T_radiation_eq[K]
        """
        atm = atmosphere_isa(altitude)
        rho = atm["rho"]
        V = mach * atm["a"]
        T_inf = atm["T"]

        # よどみ点加熱
        q_conv = self.stagnation_heating(rho, V)
        q_rad = self.radiation_heating(rho, V)
        q_total = q_conv + q_rad

        # 回復温度 (断熱壁温度)
        gamma = 1.4
        r_factor = 0.85 if mach > 0.1 else 0.0  # turbulent recovery factor
        T_recovery = T_inf * (1 + r_factor * (gamma - 1) / 2 * mach**2)

        # 輻射平衡温度 (定常)
        eps = self.tps_nose.emissivity
        if q_total > 0:
            T_rad_eq = (q_total / (eps * STEFAN_BOLTZMANN))**0.25
        else:
            T_rad_eq = T_inf

        return {
            "q_stag_conv": q_conv,
            "q_stag_rad": q_rad,
            "q_stag_total": q_total,
            "T_recovery": T_recovery,
            "T_radiation_eq": T_rad_eq,
            "V": V,
            "rho": rho,
            "T_inf": T_inf,
        }

    # ── 飛行プロファイルに沿った温度履歴 ──

    def temperature_history(self, times: np.ndarray,
                            altitudes: np.ndarray,
                            machs: np.ndarray,
                            T_init: float = 288.0) -> dict:
        """
        飛行プロファイルに沿った壁面温度の時間履歴

        Lumped capacitance (Biot < 0.1 の薄壁近似):
          ρ·cp·δ · dT/dt = q_aero - ε·σ·T⁴ - q_internal

        Args:
            times: 時刻配列 [s]
            altitudes: 高度配列 [m]
            machs: マッハ数配列
            T_init: 初期温度 [K]

        Returns:
            dict: T_nose[K], T_body[K], q_stag[W/m²], q_body[W/m²] arrays
        """
        n = len(times)
        T_nose = np.zeros(n)
        T_body = np.zeros(n)
        q_stag = np.zeros(n)
        q_body_arr = np.zeros(n)

        T_n = T_init
        T_b = T_init

        # 壁面熱容量 [J/(m²·K)]
        C_nose = self.tps_nose.rho * self.tps_nose.cp * self.tps_nose.thickness
        C_body = self.tps_body.rho * self.tps_body.cp * self.tps_body.thickness
        eps_n = self.tps_nose.emissivity
        eps_b = self.tps_body.emissivity

        for i in range(n):
            atm = atmosphere_isa(altitudes[i])
            rho = atm["rho"]
            V = machs[i] * atm["a"]

            # よどみ点加熱
            q_s = self.stagnation_heating(rho, V)
            q_stag[i] = q_s

            # ボディ加熱 (よどみ点の ~5-15%)
            q_b = q_s * 0.08
            q_body_arr[i] = q_b

            # 温度更新 (lumped capacitance)
            if i > 0:
                dt = times[i] - times[i - 1]

                # ノーズ
                q_rad_n = eps_n * STEFAN_BOLTZMANN * T_n**4
                dTdt_n = (q_s - q_rad_n) / C_nose
                T_n += dTdt_n * dt
                T_n = max(T_n, atm["T"])

                # ボディ
                q_rad_b = eps_b * STEFAN_BOLTZMANN * T_b**4
                dTdt_b = (q_b - q_rad_b) / C_body
                T_b += dTdt_b * dt
                T_b = max(T_b, atm["T"])

            T_nose[i] = T_n
            T_body[i] = T_b

        # Max-Q 加熱のピーク
        i_peak = np.argmax(q_stag)

        return {
            "time": times,
            "T_nose": T_nose,
            "T_body": T_body,
            "q_stag": q_stag,
            "q_body": q_body_arr,
            "peak_q_stag": q_stag[i_peak],
            "peak_q_time": times[i_peak],
            "max_T_nose": np.max(T_nose),
            "max_T_body": np.max(T_body),
        }

    # ── 表面温度分布 (空間) ──

    def surface_temperature_map(self, mach: float, altitude: float,
                                n_points: int = 50) -> dict:
        """
        ある飛行条件でのフェアリング表面温度分布 (定常近似)

        Args:
            mach, altitude: 飛行条件
            n_points: 表面分割数

        Returns:
            dict: x[m], r[m], q[W/m²], T_eq[K]
        """
        atm = atmosphere_isa(altitude)
        rho = atm["rho"]
        V = mach * atm["a"]

        q_stag = self.stagnation_heating(rho, V)
        x, r, theta = self.fairing.surface_points(n_points)
        q_surf = self.surface_distribution(q_stag, x, r, theta)

        # 輻射平衡温度
        eps = self.tps_nose.emissivity
        T_eq = np.zeros(n_points)
        for i in range(n_points):
            if q_surf[i] > 0:
                T_eq[i] = (q_surf[i] / (eps * STEFAN_BOLTZMANN))**0.25
            else:
                T_eq[i] = atm["T"]

        return {"x": x, "r": r, "q": q_surf, "T_eq": T_eq}

    def summary(self) -> str:
        lines = [
            "=== H3 Aerothermal Model ===",
            f"  Nose shape:      {self.fairing.shape}",
            f"  Nose tip radius: {self.fairing.nose_tip_radius:.3f} m",
            f"  Nose TPS:        {self.tps_nose.name} ({self.tps_nose.thickness*1000:.0f} mm)",
            f"  Body TPS:        {self.tps_body.name} ({self.tps_body.thickness*1000:.0f} mm)",
            f"  Nose T_max:      {self.tps_nose.T_max:.0f} K",
            f"  Body T_max:      {self.tps_body.T_max:.0f} K",
            "",
            "  Stagnation heating at typical conditions:",
        ]
        for M, alt in [(1.5, 15000), (3.0, 25000), (5.0, 40000), (7.0, 60000)]:
            r = self.heating_rate(M, alt)
            lines.append(
                f"    M={M:.1f}, h={alt/1000:.0f}km: "
                f"q={r['q_stag_conv']/1e3:.1f} kW/m², "
                f"T_eq={r['T_radiation_eq']:.0f} K"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 実行テスト
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("H3 Virtual Twin — Aerothermal Module Test")
    print("=" * 60)

    model = H3Aerothermal()
    print(model.summary())

    # Heating vs Mach/altitude
    print("\n--- Stagnation Heating vs Flight Condition ---")
    print(f"  {'M':>5s}  {'h[km]':>6s}  {'V[m/s]':>7s}  "
          f"{'q_conv[kW/m²]':>13s}  {'T_eq[K]':>8s}  {'T_rec[K]':>8s}")
    for M, alt in [(0.5, 5000), (1.0, 10000), (1.5, 15000),
                   (2.0, 20000), (3.0, 30000), (4.0, 35000),
                   (5.0, 40000), (6.0, 50000), (7.0, 60000)]:
        r = model.heating_rate(M, alt)
        print(f"  {M:5.1f}  {alt/1000:6.0f}  {r['V']:7.0f}  "
              f"{r['q_stag_conv']/1e3:13.1f}  {r['T_radiation_eq']:8.0f}  "
              f"{r['T_recovery']:8.0f}")

    # Temperature history (simplified H3-22S trajectory)
    print("\n--- Temperature History (H3-22S simplified trajectory) ---")
    t = np.linspace(0, 300, 601)
    alt = np.minimum(t * 350, 120000)
    V = np.minimum(30 + t * 22, 7500)
    a_arr = np.array([atmosphere_isa(h)["a"] for h in alt])
    M = V / a_arr

    hist = model.temperature_history(t, alt, M)
    print(f"  Peak stagnation heat flux: {hist['peak_q_stag']/1e3:.1f} kW/m²")
    print(f"  at t = {hist['peak_q_time']:.0f} s")
    print(f"  Max nose temperature:      {hist['max_T_nose']:.0f} K")
    print(f"  Max body temperature:      {hist['max_T_body']:.0f} K")
    print(f"  Nose TPS limit:            {model.tps_nose.T_max:.0f} K "
          f"{'OK' if hist['max_T_nose'] < model.tps_nose.T_max else 'EXCEEDED!'}")
    print(f"  Body TPS limit:            {model.tps_body.T_max:.0f} K "
          f"{'OK' if hist['max_T_body'] < model.tps_body.T_max else 'EXCEEDED!'}")

    # Snapshots
    print("\n  Time snapshots:")
    for ti in [0, 30, 60, 90, 120, 180, 240, 300]:
        idx = int(ti / 0.5)
        if idx < len(t):
            print(f"    t={ti:4.0f}s: M={M[idx]:.1f}, h={alt[idx]/1000:.0f}km, "
                  f"T_nose={hist['T_nose'][idx]:.0f}K, "
                  f"T_body={hist['T_body'][idx]:.0f}K, "
                  f"q={hist['q_stag'][idx]/1e3:.1f}kW/m²")

    # Surface temperature map at max-Q
    print("\n--- Surface Temperature Map at Max-Q ---")
    i_mq = np.argmax(hist["q_stag"])
    M_mq, alt_mq = M[i_mq], alt[i_mq]
    smap = model.surface_temperature_map(M_mq, alt_mq, 20)
    print(f"  Conditions: M={M_mq:.1f}, h={alt_mq/1000:.0f} km")
    print(f"  {'x[m]':>6s}  {'r[m]':>5s}  {'q[kW/m²]':>9s}  {'T_eq[K]':>8s}")
    for i in range(0, 20, 2):
        print(f"  {smap['x'][i]:6.1f}  {smap['r'][i]:5.2f}  "
              f"{smap['q'][i]/1e3:9.1f}  {smap['T_eq'][i]:8.0f}")
