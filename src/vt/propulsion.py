#!/usr/bin/env python3
"""
H3 Virtual Twin — Propulsion Module (A1-A3)

LE-9, LE-5B-3, SRB-3 のエンジンモデル。
1D サイクル解析 + エキスパンダーブリード熱力学。

References:
  - JAXA H3 Rocket User's Manual
  - JAXA LE-9 Technical Data (public)
  - Humble, Henry, Larson: "Space Propulsion Analysis and Design"
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── 物理定数 ──
G0 = 9.80665       # 標準重力加速度 [m/s²]
R_UNIV = 8314.46    # 普遍気体定数 [J/(kmol·K)]


# ═══════════════════════════════════════════════════════════════
# LE-9: LOX/LH2 Expander Bleed Cycle (1段目メインエンジン)
# ═══════════════════════════════════════════════════════════════

@dataclass
class LE9Engine:
    """
    LE-9 エキスパンダーブリードサイクル 1D モデル

    公称性能 (JAXA公開値):
      真空推力: 1,471 kN
      海面推力: ~1,220 kN
      真空Isp:  425 s
      海面Isp:  380 s
    """
    # ── 公称パラメータ ──
    name: str = "LE-9"
    thrust_vac: float = 1_471_000.0       # 真空推力 [N]
    thrust_sl: float = 1_220_000.0        # 海面推力 [N]
    isp_vac: float = 425.0               # 真空比推力 [s]
    isp_sl: float = 380.0                # 海面比推力 [s]

    # ── 推定パラメータ (CEA + 公開文献より) ──
    p_chamber: float = 12.0e6            # 燃焼室圧力 [Pa] (~120 atm)
    of_ratio: float = 5.9                # O/F 混合比 [-] (LOX/LH2)
    expansion_ratio: float = 37.0        # ノズル開口比 Ae/At [-]

    # ── 燃焼ガス特性 (CEA LOX/LH2 @ Pc=120atm, OF=5.9) ──
    gamma: float = 1.196                 # 比熱比 [-]
    T_chamber: float = 3_452.0           # 燃焼室温度 [K]
    MW_exhaust: float = 13.5             # 排気分子量 [kg/kmol]

    # ── 冷却系 (エキスパンダーブリード) ──
    coolant_bleed_fraction: float = 0.03  # ブリード率 (全流量の ~3%)
    T_coolant_in: float = 20.3           # LH2 入口温度 [K]
    T_coolant_out: float = 150.0         # LH2 出口温度 [K] (タービン前)
    p_coolant_in: float = 20.0e6         # LH2 入口圧力 [Pa]

    # ── ノズル形状 ──
    r_throat: float = 0.120              # スロート半径 [m]
    r_exit: float = 0.730                # 出口半径 [m] (from expansion_ratio)
    r_chamber: float = 0.200             # 燃焼室半径 [m]
    L_chamber: float = 0.350             # 燃焼室長さ [m]
    L_nozzle: float = 1.500              # ノズル長さ [m]

    # ── 運転範囲 ──
    throttle_min: float = 0.60           # 最低推力比 60%
    throttle_max: float = 1.00           # 最大推力比 100%

    def __post_init__(self):
        """導出パラメータを計算"""
        # 質量流量 [kg/s]
        self.mdot_total = self.thrust_vac / (self.isp_vac * G0)
        self.mdot_fuel = self.mdot_total / (1 + self.of_ratio)
        self.mdot_ox = self.mdot_total - self.mdot_fuel
        self.mdot_bleed = self.mdot_fuel * self.coolant_bleed_fraction

        # スロート面積 [m²]
        self.A_throat = math.pi * self.r_throat**2
        self.A_exit = self.A_throat * self.expansion_ratio

        # 特性排気速度 c* [m/s]
        self.c_star = self.p_chamber * self.A_throat / self.mdot_total

        # 推力係数 Cf [-]
        self.Cf_vac = self.thrust_vac / (self.p_chamber * self.A_throat)
        self.Cf_sl = self.thrust_sl / (self.p_chamber * self.A_throat)

        # 排気ガス比気体定数
        self.R_gas = R_UNIV / self.MW_exhaust  # [J/(kg·K)]

        # 理論排気速度 (等エントロピー膨張)
        self.v_exit_ideal = self._exit_velocity(self.expansion_ratio)

    def _exit_velocity(self, eps: float) -> float:
        """等エントロピー膨張による理想排気速度 [m/s]"""
        g = self.gamma
        # ノズル出口マッハ数 (面積比から反復計算)
        Me = self._mach_from_area_ratio(eps)
        # 出口圧力比
        pe_pc = (1 + (g - 1) / 2 * Me**2) ** (-g / (g - 1))
        # 排気速度
        ve = math.sqrt(
            2 * g / (g - 1) * self.R_gas * self.T_chamber
            * (1 - pe_pc ** ((g - 1) / g))
        )
        return ve

    def _mach_from_area_ratio(self, eps: float, supersonic: bool = True) -> float:
        """面積比 Ae/At からマッハ数を bisection + Newton 法で求める"""
        g = self.gamma

        def area_ratio_func(M):
            """A/A* as function of Mach"""
            t = 1 + (g - 1) / 2 * M**2
            return (1.0 / M) * (2.0 / (g + 1) * t) ** ((g + 1) / (2 * (g - 1)))

        # Bisection method (robust)
        if supersonic:
            M_lo, M_hi = 1.0001, 20.0
        else:
            M_lo, M_hi = 0.001, 0.9999

        for _ in range(100):
            M_mid = (M_lo + M_hi) / 2
            ar_mid = area_ratio_func(M_mid)
            if abs(ar_mid - eps) < 1e-10:
                return M_mid
            if ar_mid < eps:
                # Need larger area ratio → larger M (supersonic) or smaller M (subsonic)
                if supersonic:
                    M_lo = M_mid
                else:
                    M_hi = M_mid
            else:
                if supersonic:
                    M_hi = M_mid
                else:
                    M_lo = M_mid
        return (M_lo + M_hi) / 2

    def get_thrust(self, altitude: float, throttle: float = 1.0) -> dict:
        """
        高度に応じた推力を計算

        推力モデル: F = F_vac - p_atm * A_exit
        (真空推力から背圧損失を引く標準モデル)

        Args:
            altitude: 高度 [m]
            throttle: スロットル (0.6 ~ 1.0)

        Returns:
            dict with thrust[N], mdot[kg/s], isp[s], etc.
        """
        throttle = np.clip(throttle, self.throttle_min, self.throttle_max)

        # 大気圧 (指数モデル)
        p_atm = 101325.0 * math.exp(-altitude / 8500.0) if altitude < 100000 else 0.0

        # ノズル出口マッハ数・圧力
        g = self.gamma
        Me = self._mach_from_area_ratio(self.expansion_ratio)
        p_exit = self.p_chamber * (1 + (g - 1) / 2 * Me**2) ** (-g / (g - 1))

        # 推力 = F_vac - p_atm * Ae (背圧損失)
        thrust = (self.thrust_vac - p_atm * self.A_exit) * throttle
        thrust = max(thrust, 0.0)

        mdot = self.mdot_total * throttle
        isp = thrust / (mdot * G0) if mdot > 0 else 0.0

        return {
            "thrust": thrust,
            "mdot": mdot,
            "isp": isp,
            "p_exit": p_exit,
            "p_atm": p_atm,
            "Me": Me,
            "throttle": throttle,
        }

    def nozzle_contour(self, n_points: int = 200) -> tuple:
        """
        ベルノズル近似の輪郭 (x, r) を返す

        Returns:
            (x_array[m], r_array[m])
        """
        x = np.linspace(0, self.L_chamber + self.L_nozzle, n_points)
        r = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi <= self.L_chamber:
                # 燃焼室 (円筒)
                r[i] = self.r_chamber
            elif xi <= self.L_chamber + 0.05:
                # 収縮部 (円弧近似)
                t = (xi - self.L_chamber) / 0.05
                r[i] = self.r_chamber - (self.r_chamber - self.r_throat) * t**2
            elif xi <= self.L_chamber + 0.1:
                # スロート近傍
                t = (xi - self.L_chamber - 0.05) / 0.05
                r[i] = self.r_throat + (self.r_throat * 0.05) * (1 - math.cos(t * math.pi / 2))
            else:
                # 膨張部 (80% ベルノズル近似)
                t = (xi - self.L_chamber - 0.1) / (self.L_nozzle - 0.1)
                t = min(t, 1.0)
                r[i] = self.r_throat + (self.r_exit - self.r_throat) * (
                    1 - (1 - t) ** 2.5
                )
        return x, r

    def thermal_profile(self, n_points: int = 100) -> dict:
        """
        エキスパンダーブリード 1D 熱解析

        ノズル壁面に沿った:
          - ガス側壁温
          - 冷却材温度
          - 熱流束

        Returns:
            dict with arrays: x, T_gas_wall, T_coolant, q_flux, h_gas, h_coolant
        """
        x_contour, r_contour = self.nozzle_contour(n_points)
        L_total = x_contour[-1]

        # 配列初期化
        T_gas_wall = np.zeros(n_points)
        T_coolant = np.zeros(n_points)
        q_flux = np.zeros(n_points)
        h_gas = np.zeros(n_points)
        h_cool = np.zeros(n_points)

        # 冷却材は出口 → 入口 (counter-flow)
        T_c = self.T_coolant_in
        g = self.gamma

        # 壁の熱伝導率 (CuCrZr 銅合金)
        k_wall = 320.0    # [W/(m·K)]
        t_wall = 0.002     # 壁厚 [m] (2 mm)

        # 冷却チャネル
        n_channels = 360
        w_channel = 0.003   # チャネル幅 [m]
        h_channel = 0.004   # チャネル高さ [m]
        D_h = 2 * w_channel * h_channel / (w_channel + h_channel)
        A_channel = w_channel * h_channel * n_channels

        # LH2 物性 (簡易、温度依存)
        def lh2_props(T):
            """LH2 物性 (超臨界域の簡易モデル)"""
            rho = max(70.0 - 0.3 * (T - 20), 5.0)   # [kg/m³]
            cp = 10000.0 + 50 * (T - 20)              # [J/(kg·K)]
            mu = 1.3e-5 + 5e-8 * (T - 20)            # [Pa·s]
            k = 0.10 + 0.001 * (T - 20)              # [W/(m·K)]
            Pr = mu * cp / k
            return rho, cp, mu, k, Pr

        # ノズルに沿って積分 (nozzle exit → chamber: counter-flow coolant)
        for i in range(n_points):
            ri = r_contour[i]
            Ai = math.pi * ri**2

            # ローカルマッハ数
            area_ratio = Ai / self.A_throat
            if area_ratio < 1.001:
                Mi = 1.0
            elif x_contour[i] > self.L_chamber + 0.07:
                Mi = self._mach_from_area_ratio(area_ratio, supersonic=True)
            else:
                Mi = self._mach_from_area_ratio(area_ratio, supersonic=False)

            # ガス温度 (等エントロピー)
            T_local = self.T_chamber / (1 + (g - 1) / 2 * Mi**2)
            p_local = self.p_chamber * (T_local / self.T_chamber) ** (g / (g - 1))

            # 回復温度 (断熱壁温度)
            r_factor = math.sqrt(0.72)  # recovery factor (Pr^0.5)
            T_aw = T_local * (1 + r_factor * (g - 1) / 2 * Mi**2)

            # Bartz ガス側熱伝達係数 [W/(m²·K)]
            # Standard Bartz with sigma correction
            cp_gas = g * self.R_gas / (g - 1)  # [J/(kg·K)]
            mu_ref = 1.184e-5  # reference viscosity at Tc [Pa·s] (LOX/LH2)
            Pr_gas = 0.72

            # Sigma correction factor
            Tw_guess = 800.0  # wall temp guess [K]
            T_ratio = 0.5 * (Tw_guess / self.T_chamber) * (
                1 + (g - 1) / 2 * Mi**2) + 0.5
            sigma = (T_ratio ** (-0.68)) * (
                (1 + (g - 1) / 2 * Mi**2) ** (-0.12))

            # Bartz: h_g = (0.026/Dt^0.2) * (mu^0.2*cp/Pr^0.6) * (Pc*g0/c*)^0.8 * (At/A)^0.9 * sigma
            D_t = 2 * self.r_throat
            h_g = (
                0.026 / D_t**0.2
                * mu_ref**0.2 * cp_gas / Pr_gas**0.6
                * (self.p_chamber / self.c_star)**0.8
                * (self.A_throat / Ai)**0.9
                * sigma
            )
            h_gas[i] = h_g

            # 冷却材側
            rho_c, cp_c, mu_c, k_c, Pr_c = lh2_props(T_c)
            V_c = self.mdot_fuel / (rho_c * A_channel)
            Re_c = rho_c * V_c * D_h / mu_c

            # Gnielinski (Re > 2300)
            f_darcy = (0.790 * math.log(max(Re_c, 2300)) - 1.64) ** (-2)
            Nu = (f_darcy / 8 * (Re_c - 1000) * Pr_c /
                  (1 + 12.7 * math.sqrt(f_darcy / 8) * (Pr_c**(2/3) - 1)))
            Nu = max(Nu, 4.0)
            h_c = Nu * k_c / D_h
            h_cool[i] = h_c

            # 熱抵抗
            R_gas = 1.0 / h_g if h_g > 0 else 1e6
            R_wall = t_wall / k_wall
            R_cool = 1.0 / h_c if h_c > 0 else 1e6
            R_total = R_gas + R_wall + R_cool

            # 熱流束 [W/m²]
            q = (T_aw - T_c) / R_total
            q_flux[i] = q

            # ガス側壁温
            T_gas_wall[i] = T_aw - q * R_gas

            # 冷却材温度上昇 (counter-flow なので逆方向)
            if i > 0:
                dx = abs(x_contour[i] - x_contour[i - 1])
                perimeter = 2 * math.pi * ri
                dQ = q * perimeter * dx
                dT = dQ / (self.mdot_fuel * cp_c) if self.mdot_fuel * cp_c > 0 else 0
                T_c += dT

            T_coolant[i] = T_c

        return {
            "x": x_contour,
            "r": r_contour,
            "T_gas_wall": T_gas_wall,
            "T_coolant": T_coolant,
            "q_flux": q_flux,
            "h_gas": h_gas,
            "h_coolant": h_cool,
        }

    def summary(self) -> str:
        """エンジン諸元サマリー"""
        return (
            f"=== {self.name} ===\n"
            f"  Thrust (vac):    {self.thrust_vac/1e3:.0f} kN\n"
            f"  Thrust (SL):     {self.thrust_sl/1e3:.0f} kN\n"
            f"  Isp (vac):       {self.isp_vac:.0f} s\n"
            f"  Isp (SL):        {self.isp_sl:.0f} s\n"
            f"  Pc:              {self.p_chamber/1e6:.1f} MPa\n"
            f"  O/F:             {self.of_ratio:.1f}\n"
            f"  Tc:              {self.T_chamber:.0f} K\n"
            f"  mdot total:      {self.mdot_total:.1f} kg/s\n"
            f"  mdot fuel (LH2): {self.mdot_fuel:.1f} kg/s\n"
            f"  mdot ox (LOX):   {self.mdot_ox:.1f} kg/s\n"
            f"  c*:              {self.c_star:.0f} m/s\n"
            f"  Cf (vac):        {self.Cf_vac:.3f}\n"
            f"  Cf (SL):         {self.Cf_sl:.3f}\n"
            f"  Ae/At:           {self.expansion_ratio:.0f}\n"
            f"  Me (exit):       {self._mach_from_area_ratio(self.expansion_ratio):.2f}\n"
            f"  Ve (ideal):      {self.v_exit_ideal:.0f} m/s\n"
            f"  Bleed fraction:  {self.coolant_bleed_fraction*100:.1f}%\n"
        )


# ═══════════════════════════════════════════════════════════════
# LE-5B-3: LOX/LH2 Expander Bleed Cycle (2段目エンジン)
# ═══════════════════════════════════════════════════════════════

@dataclass
class LE5B3Engine:
    """LE-5B-3 (2段目エンジン) モデル"""
    name: str = "LE-5B-3"
    thrust_vac: float = 137_200.0         # 真空推力 [N]
    isp_vac: float = 448.0               # 真空比推力 [s]
    p_chamber: float = 3.6e6             # 燃焼室圧力 [Pa] (~36 atm)
    of_ratio: float = 5.0                # O/F 混合比
    gamma: float = 1.22
    T_chamber: float = 3_300.0           # [K]
    MW_exhaust: float = 12.0             # [kg/kmol]
    expansion_ratio: float = 110.0       # Ae/At
    max_burn_time: float = 740.0         # 最大燃焼時間 [s]
    throttle_min: float = 0.60
    throttle_max: float = 1.00
    restartable: bool = True             # 再着火可能

    def __post_init__(self):
        self.mdot_total = self.thrust_vac / (self.isp_vac * G0)
        self.mdot_fuel = self.mdot_total / (1 + self.of_ratio)
        self.mdot_ox = self.mdot_total - self.mdot_fuel
        self.R_gas = R_UNIV / self.MW_exhaust
        self.r_throat = math.sqrt(self.thrust_vac / (self.p_chamber * 1.8 * math.pi))
        self.A_throat = math.pi * self.r_throat**2
        self.c_star = self.p_chamber * self.A_throat / self.mdot_total
        self.Cf_vac = self.thrust_vac / (self.p_chamber * self.A_throat)

    def get_thrust(self, altitude: float = 200_000, throttle: float = 1.0) -> dict:
        """2段目は常に真空環境"""
        throttle = np.clip(throttle, self.throttle_min, self.throttle_max)
        mdot = self.mdot_total * throttle
        thrust = self.thrust_vac * throttle
        isp = self.isp_vac
        return {"thrust": thrust, "mdot": mdot, "isp": isp, "throttle": throttle}

    def summary(self) -> str:
        return (
            f"=== {self.name} ===\n"
            f"  Thrust (vac):    {self.thrust_vac/1e3:.1f} kN\n"
            f"  Isp (vac):       {self.isp_vac:.0f} s\n"
            f"  Pc:              {self.p_chamber/1e6:.1f} MPa\n"
            f"  O/F:             {self.of_ratio:.1f}\n"
            f"  mdot total:      {self.mdot_total:.1f} kg/s\n"
            f"  Max burn:        {self.max_burn_time:.0f} s\n"
            f"  Restartable:     {self.restartable}\n"
        )


# ═══════════════════════════════════════════════════════════════
# SRB-3: 固体ロケットブースター
# ═══════════════════════════════════════════════════════════════

@dataclass
class SRB3Booster:
    """
    SRB-3 固体ロケットブースター モデル

    推力プロファイルは台形近似 (実際は grain shape で決まる)
    """
    name: str = "SRB-3"
    thrust_avg: float = 2_158_000.0       # 平均推力 [N]
    isp_sl: float = 280.0                # 海面比推力 [s]
    isp_vac: float = 290.0               # 真空比推力 [s] (推定)
    burn_time: float = 113.0              # 燃焼時間 [s]
    propellant_mass: float = 66_000.0     # 推進剤質量 [kg]
    dry_mass: float = 11_000.0            # 乾燥質量 [kg]
    case_material: str = "CFRP"           # ケース素材

    # 推力プロファイル (台形): ignition → ramp-up → sustain → tail-off
    t_ignition: float = 0.0
    t_ramp_up: float = 2.0               # [s]
    t_tail_off_start: float = 105.0      # [s]
    thrust_peak_ratio: float = 1.15      # ピーク推力/平均推力

    def __post_init__(self):
        self.total_mass = self.propellant_mass + self.dry_mass
        self.mdot_avg = self.propellant_mass / self.burn_time

    def get_thrust(self, t_since_ignition: float, altitude: float = 0.0) -> dict:
        """
        点火からの経過時間に対する推力

        Args:
            t_since_ignition: SRB 点火からの時間 [s]
            altitude: 高度 [m]
        """
        t = t_since_ignition
        if t < 0 or t > self.burn_time:
            return {"thrust": 0.0, "mdot": 0.0, "isp": 0.0, "active": False}

        # 推力プロファイル (台形)
        if t < self.t_ramp_up:
            # Ramp-up
            ratio = (t / self.t_ramp_up) * self.thrust_peak_ratio
        elif t < self.t_tail_off_start:
            # Sustain (ピークから緩やかに低下)
            progress = (t - self.t_ramp_up) / (self.t_tail_off_start - self.t_ramp_up)
            ratio = self.thrust_peak_ratio - 0.2 * progress
        else:
            # Tail-off
            progress = (t - self.t_tail_off_start) / (self.burn_time - self.t_tail_off_start)
            ratio = (self.thrust_peak_ratio - 0.2) * (1 - progress**1.5)

        thrust = self.thrust_avg * max(ratio, 0.0)

        # Isp は高度で補間
        p_atm = 101325.0 * math.exp(-altitude / 8500.0) if altitude < 100000 else 0.0
        isp = self.isp_sl + (self.isp_vac - self.isp_sl) * (1 - p_atm / 101325.0)

        mdot = thrust / (isp * G0) if isp > 0 else 0.0

        return {"thrust": thrust, "mdot": mdot, "isp": isp, "active": True}

    def mass_at(self, t_since_ignition: float) -> float:
        """経過時間での残質量 [kg]"""
        t = np.clip(t_since_ignition, 0, self.burn_time)
        consumed = self.mdot_avg * t
        return self.total_mass - consumed

    def summary(self) -> str:
        return (
            f"=== {self.name} ===\n"
            f"  Thrust (avg):    {self.thrust_avg/1e3:.0f} kN\n"
            f"  Isp (SL):        {self.isp_sl:.0f} s\n"
            f"  Isp (vac):       {self.isp_vac:.0f} s\n"
            f"  Burn time:       {self.burn_time:.0f} s\n"
            f"  Propellant:      {self.propellant_mass/1e3:.0f} ton\n"
            f"  Dry mass:        {self.dry_mass/1e3:.0f} ton\n"
            f"  Case:            {self.case_material}\n"
        )


# ═══════════════════════════════════════════════════════════════
# H3 Propulsion System: 統合モデル
# ═══════════════════════════════════════════════════════════════

@dataclass
class H3PropulsionSystem:
    """
    H3 ロケット推進系統合モデル

    Configurations:
      H3-22S: LE-9 x2 + SRB-3 x2 + LE-5B-3 x1
      H3-24L: LE-9 x2 + SRB-3 x4 + LE-5B-3 x1
      H3-30S: LE-9 x3 + SRB-3 x0 + LE-5B-3 x1
    """
    config: str = "H3-22S"

    def __post_init__(self):
        configs = {
            "H3-22S": (2, 2, 1),
            "H3-24L": (2, 4, 1),
            "H3-30S": (3, 0, 1),
        }
        n_le9, n_srb, n_le5b = configs.get(self.config, (2, 2, 1))

        self.le9_engines = [LE9Engine() for _ in range(n_le9)]
        self.srb_boosters = [SRB3Booster() for _ in range(n_srb)]
        self.le5b_engine = LE5B3Engine() if n_le5b > 0 else None

        self.n_le9 = n_le9
        self.n_srb = n_srb

        # 推進剤質量
        self.s1_propellant = 209_400.0  # [kg] 1段推進剤
        self.s2_propellant = 28_000.0   # [kg] 2段推進剤
        self.s1_dry = 20_000.0          # [kg] 1段乾燥
        self.s2_dry = 4_000.0           # [kg] 2段乾燥
        self.fairing_mass = 3_400.0     # [kg]
        self.payload_mass = 4_000.0     # [kg]

    @property
    def liftoff_mass(self) -> float:
        """打上げ総質量 [kg]"""
        srb_total = sum(b.total_mass for b in self.srb_boosters)
        return (
            srb_total
            + self.s1_propellant + self.s1_dry
            + self.s2_propellant + self.s2_dry
            + self.fairing_mass + self.payload_mass
        )

    @property
    def liftoff_thrust(self) -> float:
        """離床推力 [N]"""
        le9_thrust = sum(e.thrust_sl for e in self.le9_engines)
        srb_thrust = sum(b.thrust_avg * b.thrust_peak_ratio for b in self.srb_boosters)
        return le9_thrust + srb_thrust

    @property
    def twr(self) -> float:
        """離床推力重量比"""
        return self.liftoff_thrust / (self.liftoff_mass * G0)

    def get_total_thrust(self, t: float, altitude: float,
                         s1_throttle: float = 1.0,
                         srb_active: bool = True,
                         s1_active: bool = True,
                         s2_active: bool = False,
                         s2_throttle: float = 1.0) -> dict:
        """
        時刻 t での全エンジン合計推力

        Returns:
            dict: thrust, mdot, components
        """
        total_thrust = 0.0
        total_mdot = 0.0
        components = {}

        # SRB-3
        if srb_active:
            for i, b in enumerate(self.srb_boosters):
                r = b.get_thrust(t, altitude)
                total_thrust += r["thrust"]
                total_mdot += r["mdot"]
                components[f"SRB-3_{i}"] = r

        # LE-9
        if s1_active:
            for i, e in enumerate(self.le9_engines):
                r = e.get_thrust(altitude, s1_throttle)
                total_thrust += r["thrust"]
                total_mdot += r["mdot"]
                components[f"LE-9_{i}"] = r

        # LE-5B-3
        if s2_active and self.le5b_engine:
            r = self.le5b_engine.get_thrust(altitude, s2_throttle)
            total_thrust += r["thrust"]
            total_mdot += r["mdot"]
            components["LE-5B-3"] = r

        return {
            "thrust": total_thrust,
            "mdot": total_mdot,
            "components": components,
        }

    def summary(self) -> str:
        lines = [
            f"=== H3 Propulsion System ({self.config}) ===",
            f"  Liftoff mass:  {self.liftoff_mass/1e3:.0f} ton",
            f"  Liftoff thrust: {self.liftoff_thrust/1e3:.0f} kN",
            f"  T/W ratio:     {self.twr:.2f}",
            f"  LE-9:  {self.n_le9} engines",
            f"  SRB-3: {self.n_srb} boosters",
            f"  LE-5B-3: {'yes' if self.le5b_engine else 'no'}",
            "",
        ]
        for e in self.le9_engines[:1]:
            lines.append(e.summary())
        for b in self.srb_boosters[:1]:
            lines.append(b.summary())
        if self.le5b_engine:
            lines.append(self.le5b_engine.summary())
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 実行テスト
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("H3 Virtual Twin — Propulsion Module Test")
    print("=" * 60)

    # H3-22S
    h3 = H3PropulsionSystem("H3-22S")
    print(h3.summary())

    # 推力 vs 高度プロファイル
    print("\n--- Thrust vs Altitude (LE-9 single engine) ---")
    le9 = h3.le9_engines[0]
    for alt in [0, 10000, 30000, 50000, 100000, 200000]:
        r = le9.get_thrust(alt)
        print(f"  h={alt/1000:6.0f} km: F={r['thrust']/1e3:7.0f} kN, "
              f"Isp={r['isp']:.0f} s, Pe={r['p_exit']/1e3:.0f} kPa")

    # SRB 推力プロファイル
    print("\n--- SRB-3 Thrust Profile ---")
    srb = h3.srb_boosters[0]
    for t in [0, 1, 5, 50, 100, 110, 113, 114]:
        r = srb.get_thrust(t, altitude=t * 400)
        print(f"  t={t:4.0f}s: F={r['thrust']/1e3:7.0f} kN, "
              f"Isp={r['isp']:.0f} s, active={r['active']}")

    # 全段推力 (離床)
    print("\n--- Total Thrust at Liftoff ---")
    r = h3.get_total_thrust(t=0, altitude=0)
    print(f"  Total: {r['thrust']/1e3:.0f} kN ({r['thrust']/1e6:.1f} MN)")
    print(f"  mdot:  {r['mdot']:.0f} kg/s")

    # LE-9 熱解析
    print("\n--- LE-9 Thermal Profile (1D Expander Bleed) ---")
    thermal = le9.thermal_profile(50)
    peak_q = np.max(thermal["q_flux"])
    peak_Tw = np.max(thermal["T_gas_wall"])
    T_coolant_out = thermal["T_coolant"][-1]
    print(f"  Peak heat flux:     {peak_q/1e6:.1f} MW/m²")
    print(f"  Peak wall temp:     {peak_Tw:.0f} K")
    print(f"  Coolant outlet:     {T_coolant_out:.0f} K")
    print(f"  Throat position:    x={thermal['x'][np.argmax(thermal['q_flux'])]:.3f} m")

    # H3-30S (SRB なし)
    print("\n" + "=" * 60)
    h3_30 = H3PropulsionSystem("H3-30S")
    print(h3_30.summary())
