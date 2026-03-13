#!/usr/bin/env python3
"""
H3 Virtual Twin — External Aerodynamics Module (A2)

H3 ロケットの外部空力特性テーブル。
Cd, Cn, Cp (圧力中心) を Mach 数 × 迎角 α でテーブル化。

手法:
  - ゼロ揚力抗力: 摩擦抗力 (Van Driest II) + 遷音速波動抗力 (Haack body)
  - 法線力: 修正スレンダーボディ理論 (Modified Newtonian + crossflow)
  - 基底抗力: 経験式 (base drag correlation)

H3 形状:
  全長 ~63 m, コア直径 5.2 m, フェアリング長 15 m
  ノーズコーン: 楕円型 (fineness ratio ~3.0)
  SRB 付き形態ではブースター間の干渉抗力を補正

References:
  - Barrowman, "The Practical Calculation of the Aerodynamic Characteristics
    of Slender Finless Vehicles" (1967)
  - NASA SP-8020 "Surface Pressure on Pointed Bodies of Revolution"
  - Missile DATCOM methodology
  - JAXA H3 Rocket User's Manual (geometry)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# H3 外部形状定義
# ═══════════════════════════════════════════════════════════════

@dataclass
class H3Geometry:
    """H3 ロケット外部形状パラメータ"""
    # 全体
    length_total: float = 63.0          # 全長 [m]
    diameter: float = 5.2               # コア直径 [m]

    # フェアリング (ノーズコーン)
    fairing_length: float = 15.0        # フェアリング長 [m]
    fairing_nose_length: float = 8.0    # ノーズ先端部長さ [m]
    fairing_cylinder_length: float = 7.0  # 円筒部長さ [m]
    nose_fineness: float = 3.077        # ノーズ形状比 L_nose/D (8.0/2.6)

    # 1段目
    s1_length: float = 37.0             # 1段長さ [m]

    # 2段目
    s2_length: float = 11.0             # 2段長さ [m]

    # SRB-3
    srb_length: float = 14.6            # SRB 長さ [m]
    srb_diameter: float = 2.5           # SRB 直径 [m]

    def __post_init__(self):
        self.radius = self.diameter / 2
        # 基準面積 (コア断面積)
        self.S_ref = math.pi / 4 * self.diameter**2  # ~21.24 m²
        # 濡れ面積 (近似: 円筒 + ノーズ)
        self.S_wet_body = (
            math.pi * self.diameter * (self.length_total - self.fairing_nose_length)
            + math.pi * self.radius * math.sqrt(
                self.radius**2 + self.fairing_nose_length**2)
        )
        # SRB 濡れ面積 (1本あたり)
        self.S_wet_srb = math.pi * self.srb_diameter * self.srb_length


# ═══════════════════════════════════════════════════════════════
# 大気モデル (ISA 簡易)
# ═══════════════════════════════════════════════════════════════

def atmosphere_isa(altitude: float) -> dict:
    """
    国際標準大気 (ISA) — 簡易モデル

    Args:
        altitude: 高度 [m]

    Returns:
        dict: T[K], p[Pa], rho[kg/m³], a[m/s], mu[Pa·s]
    """
    if altitude < 11000:
        # 対流圏
        T = 288.15 - 0.0065 * altitude
        p = 101325.0 * (T / 288.15) ** 5.2561
    elif altitude < 20000:
        # 成層圏下部 (等温)
        T = 216.65
        p = 22632.0 * math.exp(-0.0001577 * (altitude - 11000))
    elif altitude < 32000:
        # 成層圏上部
        T = 216.65 + 0.001 * (altitude - 20000)
        p = 5474.9 * (T / 216.65) ** (-34.163)
    elif altitude < 47000:
        T = 228.65 + 0.0028 * (altitude - 32000)
        p = 868.02 * (T / 228.65) ** (-12.201)
    elif altitude < 100000:
        T = max(165.0, 270.65 - 0.002 * (altitude - 47000))
        p = 110.91 * math.exp(-0.000126 * (altitude - 47000))
    else:
        T = 200.0
        p = max(0.01, 0.37 * math.exp(-0.00012 * (altitude - 100000)))

    rho = p / (287.058 * T) if T > 0 else 0.0
    a = math.sqrt(1.4 * 287.058 * T) if T > 0 else 300.0
    # Sutherland の粘性法則
    mu = 1.458e-6 * T**1.5 / (T + 110.4)

    return {"T": T, "p": p, "rho": rho, "a": a, "mu": mu}


# ═══════════════════════════════════════════════════════════════
# H3 空力テーブル
# ═══════════════════════════════════════════════════════════════

@dataclass
class H3Aerodynamics:
    """
    H3 ロケット外部空力モデル

    Mach × α のテーブル補間で Cd, Cn, Cp を返す。
    テーブルはスレンダーボディ理論 + 経験式で事前生成。

    Usage:
        aero = H3Aerodynamics()
        forces = aero.get_forces(mach=1.2, alpha_deg=2.0, altitude=30000)
    """
    config: str = "H3-22S"
    geometry: H3Geometry = field(default_factory=H3Geometry)

    # Mach 数テーブル (亜音速 ~ 極超音速)
    _mach_table: np.ndarray = field(default=None, repr=False)
    # 迎角テーブル [deg]
    _alpha_table: np.ndarray = field(default=None, repr=False)
    # 空力係数テーブル
    _Cd0_table: np.ndarray = field(default=None, repr=False)
    _Cna_table: np.ndarray = field(default=None, repr=False)
    _Xcp_table: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        n_srb = {"H3-22S": 2, "H3-24L": 4, "H3-30S": 0}.get(self.config, 2)
        self.n_srb = n_srb
        self._build_tables()

    def _build_tables(self):
        """空力係数テーブルを半経験式で構築"""
        # Mach: 0 ~ 8.0 (40 points, 遷音速域は密に)
        mach_sub = np.linspace(0.0, 0.8, 9)
        mach_trans = np.linspace(0.85, 1.3, 10)
        mach_super = np.linspace(1.5, 8.0, 14)
        self._mach_table = np.concatenate([mach_sub, mach_trans, mach_super])

        # α: 0 ~ 10 deg (11 points)
        self._alpha_table = np.linspace(0, 10, 11)

        n_m = len(self._mach_table)
        n_a = len(self._alpha_table)

        self._Cd0_table = np.zeros((n_m, n_a))
        self._Cna_table = np.zeros((n_m, n_a))
        self._Xcp_table = np.zeros((n_m, n_a))

        for i, M in enumerate(self._mach_table):
            for j, alpha in enumerate(self._alpha_table):
                self._Cd0_table[i, j] = self._calc_cd0(M, alpha)
                self._Cna_table[i, j] = self._calc_cn(M, alpha)
                self._Xcp_table[i, j] = self._calc_xcp(M, alpha)

    # ── ゼロ揚力抗力 Cd0 ──

    def _calc_cd0(self, M: float, alpha_deg: float) -> float:
        """
        ゼロ揚力抗力係数 Cd0

        = Cd_friction + Cd_wave + Cd_base + Cd_interference + Cd_induced
        """
        cd_f = self._cd_friction(M)
        cd_w = self._cd_wave(M)
        cd_b = self._cd_base(M)
        cd_i = self._cd_interference(M)
        cd_ind = self._cd_induced(M, alpha_deg)
        return cd_f + cd_w + cd_b + cd_i + cd_ind

    def _cd_friction(self, M: float) -> float:
        """摩擦抗力 (turbulent flat plate + compressibility correction)"""
        # Re at sea level, V~M*340
        L = self.geometry.length_total
        V = max(M * 340, 1.0)
        # Typical Re ~ 1e8 for launch vehicles
        Re = 1.225 * V * L / 1.8e-5
        Re = max(Re, 1e6)

        # Schlichting (turbulent, incompressible)
        Cf_inc = 0.455 / (math.log10(Re))**2.58

        # Van Driest II 圧縮性補正
        if M < 0.1:
            Cf = Cf_inc
        else:
            # 簡易: Cf/Cf_inc ≈ 1 / (1 + 0.144 * M²)^0.65
            Cf = Cf_inc / (1 + 0.144 * M**2)**0.65

        # 濡れ面積比
        S_wet = self.geometry.S_wet_body
        S_wet += self.n_srb * self.geometry.S_wet_srb
        Cd_f = Cf * S_wet / self.geometry.S_ref

        return Cd_f

    def _cd_wave(self, M: float) -> float:
        """遷音速波動抗力 (transonic drag rise)"""
        if M < 0.8:
            return 0.0
        elif M < 1.0:
            # Drag rise: cubic interpolation
            x = (M - 0.8) / 0.2
            return 0.12 * x**3
        elif M <= 1.2:
            # Peak drag at M~1.05
            return 0.12 * (1.0 - 2.0 * (M - 1.05)**2 / 0.15**2)
        else:
            # Supersonic wave drag decay: Prandtl-Glauert inspired
            beta = math.sqrt(M**2 - 1)
            # Haack body minimum drag
            n = self.geometry.nose_fineness
            return 0.08 / (beta * (1 + 0.2 * n))

    def _cd_base(self, M: float) -> float:
        """基底抗力 (engine nozzle base area)"""
        # Base area ratio (nozzle area / reference area)
        # H3: ~30% of base area is nozzle exit
        A_base_ratio = 0.30
        if M < 0.6:
            Cd_b = 0.12 + 0.13 * M**2
        elif M < 1.0:
            Cd_b = 0.25 - 0.1 * (M - 0.8)**2
        elif M < 2.0:
            Cd_b = 0.25 / M
        else:
            Cd_b = 0.15 / M
        return Cd_b * A_base_ratio

    def _cd_interference(self, M: float) -> float:
        """SRB 干渉抗力"""
        if self.n_srb == 0:
            return 0.0
        # SRB-コア間の干渉抗力 (遷音速域でピーク)
        cd_int_base = 0.015 * self.n_srb
        if M < 0.8:
            return cd_int_base * 0.5
        elif M < 1.3:
            x = (M - 0.8) / 0.5
            return cd_int_base * (0.5 + 0.5 * math.sin(x * math.pi / 2))
        else:
            beta = math.sqrt(max(M**2 - 1, 0.01))
            return cd_int_base * 0.8 / beta

    def _cd_induced(self, M: float, alpha_deg: float) -> float:
        """誘導抗力 (迎角による追加抗力)"""
        alpha = math.radians(alpha_deg)
        if alpha < 0.001:
            return 0.0
        # Crossflow drag on cylinder body
        Cd_cross = 1.2  # cylinder crossflow Cd
        S_cross = self.geometry.diameter * self.geometry.length_total
        S_cross += self.n_srb * self.geometry.srb_diameter * self.geometry.srb_length
        cd_ind = Cd_cross * (S_cross / self.geometry.S_ref) * math.sin(alpha)**2 * math.cos(alpha)
        return cd_ind

    # ── 法線力係数 Cn ──

    def _calc_cn(self, M: float, alpha_deg: float) -> float:
        """
        法線力係数 Cn

        Cn = Cnα * α + Cn_crossflow
        (スレンダーボディ理論 + crossflow 補正)
        """
        alpha = math.radians(alpha_deg)
        if abs(alpha) < 1e-6:
            return 0.0

        # Slender body Cnα [per rad]
        # Cn_alpha = 2 (slender body theory, per rad)
        Cn_alpha = 2.0

        # 圧縮性補正
        if M < 0.8:
            beta_corr = 1.0
        elif M < 1.05:
            beta_corr = 1.0 / math.sqrt(max(1 - M**2, 0.01))
            beta_corr = min(beta_corr, 3.0)  # cap at transonic
        else:
            beta_corr = 1.0 / math.sqrt(M**2 - 1)
            beta_corr = min(beta_corr, 3.0)

        # Potential flow (linear)
        Cn_pot = Cn_alpha * beta_corr * alpha

        # Crossflow (viscous, high-alpha correction)
        Cd_cross = 1.2
        S_cross_ratio = (self.geometry.diameter * self.geometry.length_total
                         / self.geometry.S_ref)
        Cn_cross = Cd_cross * S_cross_ratio * math.sin(alpha)**2

        # SRB 寄与
        if self.n_srb > 0:
            srb_ratio = (self.geometry.srb_diameter * self.geometry.srb_length
                         * self.n_srb / self.geometry.S_ref)
            Cn_cross += Cd_cross * srb_ratio * math.sin(alpha)**2 * 0.5

        Cn = Cn_pot + Cn_cross
        return Cn

    # ── 圧力中心 Xcp ──

    def _calc_xcp(self, M: float, alpha_deg: float) -> float:
        """
        圧力中心位置 Xcp (ノーズ先端からの距離 / 全長)

        Returns:
            Xcp/L_total [-] (0 = ノーズ先端, 1 = 基底)
        """
        L = self.geometry.length_total
        L_nose = self.geometry.fairing_nose_length

        # ノーズ寄与 (楕円型: Xcp_nose = 0.466 * L_nose from tip)
        Xcp_nose = 0.466 * L_nose

        # ボディ (円筒): Xcp_body = L_nose + 0.5 * L_body
        L_body = L - L_nose
        Xcp_body = L_nose + 0.5 * L_body

        # ノーズ / ボディの法線力比率 (Mach 依存)
        alpha = math.radians(max(alpha_deg, 0.1))
        # Potential flow: nose dominates at low alpha
        # Crossflow: body dominates at high alpha
        ratio_nose = 0.3 / (1 + 0.5 * alpha_deg)  # nose fraction decreases with alpha

        # 遷音速ではやや前方にシフト
        if 0.8 < M < 1.3:
            ratio_nose *= 1.1

        Xcp = ratio_nose * Xcp_nose + (1 - ratio_nose) * Xcp_body
        return Xcp / L

    # ── テーブル補間 ──

    def _interp2d(self, table: np.ndarray, M: float, alpha_deg: float) -> float:
        """2D 線形補間"""
        M = np.clip(M, self._mach_table[0], self._mach_table[-1])
        alpha_deg = np.clip(alpha_deg, self._alpha_table[0], self._alpha_table[-1])

        # Mach 軸インデックス
        i = np.searchsorted(self._mach_table, M) - 1
        i = np.clip(i, 0, len(self._mach_table) - 2)
        # Alpha 軸インデックス
        j = np.searchsorted(self._alpha_table, alpha_deg) - 1
        j = np.clip(j, 0, len(self._alpha_table) - 2)

        # 重み
        M0, M1 = self._mach_table[i], self._mach_table[i + 1]
        a0, a1 = self._alpha_table[j], self._alpha_table[j + 1]
        wm = (M - M0) / (M1 - M0) if M1 > M0 else 0.0
        wa = (alpha_deg - a0) / (a1 - a0) if a1 > a0 else 0.0

        # Bilinear
        v00 = table[i, j]
        v10 = table[i + 1, j]
        v01 = table[i, j + 1]
        v11 = table[i + 1, j + 1]

        v = (v00 * (1 - wm) * (1 - wa) +
             v10 * wm * (1 - wa) +
             v01 * (1 - wm) * wa +
             v11 * wm * wa)
        return float(v)

    # ── Public API ──

    def get_coefficients(self, mach: float, alpha_deg: float) -> dict:
        """
        空力係数を返す

        Args:
            mach: マッハ数
            alpha_deg: 迎角 [deg]

        Returns:
            dict: Cd, Cn, Xcp_norm (Xcp/L)
        """
        Cd = self._interp2d(self._Cd0_table, mach, abs(alpha_deg))
        Cn = self._interp2d(self._Cna_table, mach, abs(alpha_deg))
        if alpha_deg < 0:
            Cn = -Cn
        Xcp = self._interp2d(self._Xcp_table, mach, abs(alpha_deg))
        return {"Cd": Cd, "Cn": Cn, "Xcp_norm": Xcp}

    def get_forces(self, mach: float, alpha_deg: float,
                   altitude: float) -> dict:
        """
        空力 (ドラッグ, リフト, モーメント) を計算

        Flight Orchestrator 向けインターフェース。

        Args:
            mach: マッハ数
            alpha_deg: 迎角 [deg]
            altitude: 高度 [m]

        Returns:
            dict: drag[N], lift[N], normal[N], moment[Nm],
                  Cd, Cn, q_dyn[Pa], Xcp[m]
        """
        atm = atmosphere_isa(altitude)
        rho = atm["rho"]
        a = atm["a"]

        V = mach * a
        q = 0.5 * rho * V**2  # 動圧 [Pa]

        coeff = self.get_coefficients(mach, alpha_deg)
        Cd = coeff["Cd"]
        Cn = coeff["Cn"]
        Xcp_norm = coeff["Xcp_norm"]

        S = self.geometry.S_ref

        # 機軸座標系
        alpha = math.radians(alpha_deg)
        # Axial force (drag direction) and Normal force
        # Ca = Cd * cos(α) - Cn * sin(α)  (axial)
        # CN_total = Cd * sin(α) + Cn * cos(α)  (normal total)
        # ≈ Cd, Cn for small alpha
        drag = q * S * Cd       # 抗力 [N] (velocity direction)
        normal = q * S * Cn     # 法線力 [N]

        # Lift and drag in velocity frame
        lift = normal * math.cos(alpha) - drag * math.sin(alpha) if abs(alpha) > 1e-6 else 0.0
        drag_total = drag * math.cos(alpha) + normal * math.sin(alpha) if abs(alpha) > 1e-6 else drag

        # ピッチングモーメント (重心まわり)
        L = self.geometry.length_total
        Xcp = Xcp_norm * L  # ノーズ先端からの距離 [m]
        Xcg = 0.55 * L      # 重心位置 (打上げ時、概算)
        moment_arm = Xcp - Xcg
        moment = normal * moment_arm  # [Nm]

        return {
            "drag": drag_total,
            "lift": lift,
            "normal": normal,
            "moment": moment,
            "Cd": Cd,
            "Cn": Cn,
            "q_dyn": q,
            "Xcp": Xcp,
            "Xcg": Xcg,
            "V": V,
            "rho": rho,
        }

    def max_q_profile(self, altitudes: np.ndarray, machs: np.ndarray,
                      alpha_deg: float = 0.0) -> dict:
        """
        飛行プロファイルに沿った動圧・空力荷重の計算

        Args:
            altitudes: 高度配列 [m]
            machs: マッハ数配列
            alpha_deg: 迎角 [deg]

        Returns:
            dict: q_dyn[Pa], drag[N], Cd arrays + max_q info
        """
        n = len(altitudes)
        q_arr = np.zeros(n)
        drag_arr = np.zeros(n)
        cd_arr = np.zeros(n)

        for i in range(n):
            r = self.get_forces(machs[i], alpha_deg, altitudes[i])
            q_arr[i] = r["q_dyn"]
            drag_arr[i] = r["drag"]
            cd_arr[i] = r["Cd"]

        i_max = np.argmax(q_arr)
        return {
            "q_dyn": q_arr,
            "drag": drag_arr,
            "Cd": cd_arr,
            "max_q": q_arr[i_max],
            "max_q_altitude": altitudes[i_max],
            "max_q_mach": machs[i_max],
            "max_q_index": i_max,
        }

    def summary(self) -> str:
        """空力モデルサマリー"""
        lines = [
            f"=== H3 Aerodynamics ({self.config}) ===",
            f"  Reference area:  {self.geometry.S_ref:.2f} m²",
            f"  Body length:     {self.geometry.length_total:.0f} m",
            f"  Core diameter:   {self.geometry.diameter:.1f} m",
            f"  SRB count:       {self.n_srb}",
            f"  Mach range:      {self._mach_table[0]:.1f} - {self._mach_table[-1]:.1f}",
            f"  Alpha range:     {self._alpha_table[0]:.0f} - {self._alpha_table[-1]:.0f} deg",
            "",
            "  Key Cd values (α=0°):",
        ]
        for M in [0.3, 0.8, 1.0, 1.2, 2.0, 4.0]:
            c = self.get_coefficients(M, 0.0)
            lines.append(f"    M={M:.1f}: Cd={c['Cd']:.4f}")

        lines.append("")
        lines.append("  Key Cn values (M=1.2):")
        for a in [0, 2, 5, 10]:
            c = self.get_coefficients(1.2, a)
            lines.append(f"    α={a:2d}°: Cn={c['Cn']:.4f}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 実行テスト
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("H3 Virtual Twin — Aerodynamics Module Test")
    print("=" * 60)

    aero = H3Aerodynamics("H3-22S")
    print(aero.summary())

    # Cd vs Mach (α=0°)
    print("\n--- Cd vs Mach (α=0°) ---")
    for M in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2,
              1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        c = aero.get_coefficients(M, 0.0)
        print(f"  M={M:5.2f}: Cd={c['Cd']:.4f}")

    # Cn vs alpha (M=1.2)
    print("\n--- Cn vs Alpha (M=1.2) ---")
    for a in range(11):
        c = aero.get_coefficients(1.2, a)
        print(f"  α={a:2d}°: Cn={c['Cn']:.4f}, Xcp/L={c['Xcp_norm']:.3f}")

    # Aerodynamic forces at typical flight conditions
    print("\n--- Forces at Typical Flight Points ---")
    conditions = [
        ("Liftoff",   0.0, 0.05, 0),
        ("Subsonic",  10000, 0.8, 30000),
        ("Max-Q",     30000, 1.2, 12000),
        ("Transonic", 40000, 1.5, 15000),
        ("Supersonic", 50000, 3.0, 25000),
        ("High alt",  80000, 6.0, 50000),
    ]
    for name, alt, M, _ in conditions:
        r = aero.get_forces(M, 2.0, alt)
        print(f"  {name:12s}: M={M:.1f}, h={alt/1000:.0f}km, "
              f"q={r['q_dyn']/1e3:.1f}kPa, "
              f"Drag={r['drag']/1e3:.1f}kN, Cd={r['Cd']:.4f}")

    # Typical H3-22S trajectory (simplified)
    print("\n--- Max-Q Analysis (simplified trajectory) ---")
    t = np.linspace(0, 300, 301)
    # Rough trajectory: linear altitude + velocity ramp
    alt = np.minimum(t * 400, 120000)  # ~400 m/s climb rate
    V = np.minimum(30 + t * 25, 7500)  # acceleration
    a_sound = np.array([atmosphere_isa(h)["a"] for h in alt])
    M = V / a_sound

    result = aero.max_q_profile(alt, M)
    print(f"  Max dynamic pressure: {result['max_q']/1e3:.1f} kPa")
    print(f"  at altitude:          {result['max_q_altitude']/1e3:.1f} km")
    print(f"  at Mach:              {result['max_q_mach']:.2f}")

    # H3-30S (no SRB)
    print("\n" + "=" * 60)
    aero_30 = H3Aerodynamics("H3-30S")
    print(aero_30.summary())
