#!/usr/bin/env python3
"""
H3 Virtual Twin — Flight Orchestrator (B1-B4)

全サブシステムを時系列で統合する Flight Orchestrator。
T-0 (離床) → SECO (2段エンジン停止) まで一気通貫のシミュレーション。

飛行フェーズ:
  P1: SRB+S1      T+0~116s    LE-9×2 + SRB-3×2, 重力ターン
  SRB Sep          T+116s      SRB 分離 (衝撃外乱)
  P2: S1 only      T+116~207s  LE-9×2, 大気圏内
  Fairing Sep      T+207s      フェアリング分離 (質量変化)
  P3: S1 coast     T+207~298s  LE-9×2, 大気圏外
  MECO/Stage Sep   T+298s      1段停止 → 段間分離
  P4: S2 burn      T+303~850s  LE-5B-3 (真空)
  SECO             T+850s      軌道投入

統合ループ:
  1. イベント判定
  2. 推進力 (propulsion)
  3. 空力 (aerodynamics)
  4. GNC / 姿勢制御 (attitude_control)
  5. 3DOF 運動方程式積分
  6. 空力加熱 (aerothermal)
  7. テレメトリ記録

References:
  - JAXA H3 Rocket User's Manual
  - wiki_repo/H3-Virtual-Twin.md
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from .propulsion import H3PropulsionSystem, G0
from .aerodynamics import H3Aerodynamics, atmosphere_isa
from .aerothermal import H3Aerothermal
from .attitude_control import H3AttitudeController, AttitudeState, GravityTurnProfile


# ── 定数 ──
RE = 6_371_000.0    # 地球半径 [m]
MU = 3.986004e14    # 地球重力定数 [m³/s²]


# ═══════════════════════════════════════════════════════════════
# 飛行状態ベクトル
# ═══════════════════════════════════════════════════════════════

@dataclass
class FlightState:
    """飛行状態ベクトル (3DOF position + attitude)"""
    # 時刻
    t: float = 0.0                  # [s]

    # 位置・速度 (地表固定座標: x=downrange, y=cross-range, z=altitude)
    x: float = 0.0                  # ダウンレンジ [m]
    y: float = 0.0                  # クロスレンジ [m]
    z: float = 0.0                  # 高度 [m]
    vx: float = 0.0                # [m/s]
    vy: float = 0.0                # [m/s]
    vz: float = 0.0                # [m/s]

    # 姿勢 (オイラー角)
    pitch: float = math.radians(90)  # [rad] 鉛直=90°
    yaw: float = 0.0                # [rad]
    roll: float = 0.0               # [rad]
    pitch_rate: float = 0.0         # [rad/s]
    yaw_rate: float = 0.0           # [rad/s]
    roll_rate: float = 0.0          # [rad/s]

    # 質量
    mass: float = 0.0              # 現在質量 [kg]

    # 飛行パラメータ (導出)
    speed: float = 0.0             # [m/s]
    mach: float = 0.0
    alpha: float = 0.0             # 迎角 [rad]
    beta: float = 0.0              # 横滑り角 [rad]
    q_dyn: float = 0.0             # 動圧 [Pa]
    gamma: float = math.radians(90)  # 飛行経路角 [rad]
    g_local: float = G0            # 局所重力加速度 [m/s²]
    accel: float = 0.0             # 加速度 [m/s²]
    accel_axial: float = 0.0       # 軸方向加速度 [G]

    # フェーズ
    phase: str = "P1_SRB_S1"

    @property
    def altitude(self) -> float:
        return self.z

    def velocity_magnitude(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def update_derived(self):
        """導出パラメータを更新"""
        self.speed = self.velocity_magnitude()
        r = RE + self.z
        self.g_local = MU / r**2

        atm = atmosphere_isa(self.z)
        a = atm["a"]
        rho = atm["rho"]

        self.mach = self.speed / a if a > 0 else 0.0
        self.q_dyn = 0.5 * rho * self.speed**2

        # 飛行経路角
        if self.speed > 1.0:
            self.gamma = math.atan2(self.vz, max(self.vx, 0.01))
        # 迎角 (ピッチ角 - 飛行経路角)
        self.alpha = self.pitch - self.gamma


# ═══════════════════════════════════════════════════════════════
# 飛行イベント
# ═══════════════════════════════════════════════════════════════

@dataclass
class FlightEvent:
    """飛行イベント定義"""
    name: str
    time: float                     # 発生時刻 [s]
    triggered: bool = False
    description: str = ""

    # イベント発生条件 (時刻 or 条件)
    condition: Optional[Callable] = None  # state → bool

    def check(self, t: float, state: FlightState) -> bool:
        """イベント発生判定"""
        if self.triggered:
            return False
        if self.condition is not None:
            return self.condition(state)
        return t >= self.time


@dataclass
class EventManager:
    """飛行イベント管理"""
    events: List[FlightEvent] = field(default_factory=list)
    log: List[tuple] = field(default_factory=list)

    def add(self, name: str, time: float, description: str = "",
            condition: Callable = None):
        self.events.append(FlightEvent(name, time, description=description,
                                       condition=condition))

    def check(self, t: float, state: FlightState) -> List[str]:
        """全イベントをチェックし、発火したイベント名リストを返す"""
        fired = []
        for event in self.events:
            if event.check(t, state):
                event.triggered = True
                fired.append(event.name)
                self.log.append((t, event.name, event.description))
        return fired


# ═══════════════════════════════════════════════════════════════
# テレメトリレコーダー
# ═══════════════════════════════════════════════════════════════

class TelemetryRecorder:
    """飛行データ記録"""

    FIELDS = [
        "t", "x", "z", "vx", "vz", "speed", "mach", "altitude",
        "mass", "q_dyn", "gamma_deg", "pitch_deg", "alpha_deg",
        "thrust", "drag", "accel_G", "accel_axial_G",
        "gimbal_pitch", "T_nose", "T_body", "q_stag",
        "phase",
    ]

    def __init__(self):
        self._data = {k: [] for k in self.FIELDS}

    def record(self, state: FlightState, extras: dict = None):
        """1タイムステップのデータを記録"""
        self._data["t"].append(state.t)
        self._data["x"].append(state.x)
        self._data["z"].append(state.z)
        self._data["vx"].append(state.vx)
        self._data["vz"].append(state.vz)
        self._data["speed"].append(state.speed)
        self._data["mach"].append(state.mach)
        self._data["altitude"].append(state.z)
        self._data["mass"].append(state.mass)
        self._data["q_dyn"].append(state.q_dyn)
        self._data["gamma_deg"].append(math.degrees(state.gamma))
        self._data["pitch_deg"].append(math.degrees(state.pitch))
        self._data["alpha_deg"].append(math.degrees(state.alpha))
        self._data["phase"].append(state.phase)

        if extras:
            for k in ["thrust", "drag", "accel_G", "accel_axial_G",
                       "gimbal_pitch", "T_nose", "T_body", "q_stag"]:
                self._data[k].append(extras.get(k, 0.0))
        else:
            for k in ["thrust", "drag", "accel_G", "accel_axial_G",
                       "gimbal_pitch", "T_nose", "T_body", "q_stag"]:
                self._data[k].append(0.0)

    def to_arrays(self) -> dict:
        """numpy 配列に変換"""
        result = {}
        for k, v in self._data.items():
            if k == "phase":
                result[k] = v
            else:
                result[k] = np.array(v)
        return result


# ═══════════════════════════════════════════════════════════════
# H3 Flight Orchestrator
# ═══════════════════════════════════════════════════════════════

@dataclass
class H3FlightOrchestrator:
    """
    H3 ロケット統合シミュレーション — Flight Orchestrator

    全サブシステムを時系列で統合し、T-0 → SECO の飛行を再現。

    Usage:
        orch = H3FlightOrchestrator("H3-22S")
        result = orch.run_mission()
        print(orch.mission_summary(result))
    """
    config: str = "H3-22S"
    dt: float = 0.1                 # 時間ステップ [s]

    # サブシステム
    propulsion: H3PropulsionSystem = None
    aero: H3Aerodynamics = None
    thermal: H3Aerothermal = None
    attitude: H3AttitudeController = None

    # イベント管理
    events: EventManager = field(default_factory=EventManager)
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)

    # イベント時刻 [s]
    t_srb_sep: float = 116.0
    t_fairing_sep: float = 230.0
    t_meco: float = 298.0
    t_stage_sep: float = 303.0
    t_s2_ignition: float = 308.0
    t_seco: float = 850.0

    # SRB 分離外乱
    srb_sep_impulse_pitch: float = 0.3    # [deg/s] 姿勢外乱
    stage_sep_impulse_pitch: float = 0.2  # [deg/s]

    def __post_init__(self):
        self._init_subsystems()
        self._init_events()
        # フェーズフラグ
        self._srb_active = True
        self._s1_active = True
        self._s2_active = False
        self._fairing_on = True

    def _init_subsystems(self):
        if self.propulsion is None:
            self.propulsion = H3PropulsionSystem(self.config)
        if self.aero is None:
            self.aero = H3Aerodynamics(self.config)
        if self.thermal is None:
            self.thermal = H3Aerothermal()
        if self.attitude is None:
            profile = GravityTurnProfile(
                t_srb_sep=self.t_srb_sep,
                t_fairing_sep=self.t_fairing_sep,
                t_meco=self.t_meco,
                t_s2_ignition=self.t_s2_ignition,
                t_seco=self.t_seco,
            )
            self.attitude = H3AttitudeController(self.config, profile=profile)

    def _init_events(self):
        self.events = EventManager()
        self.events.add("SRB_BURNOUT", self.t_srb_sep - 3,
                        "SRB-3 burnout (tail-off)")
        self.events.add("SRB_SEP", self.t_srb_sep,
                        "SRB-3 jettison")
        self.events.add("FAIRING_SEP", self.t_fairing_sep,
                        "Fairing jettison (h > 120 km, q < 1 kPa)")
        self.events.add("MECO", self.t_meco,
                        "Main Engine Cut-Off")
        self.events.add("STAGE_SEP", self.t_stage_sep,
                        "Stage 1/2 separation")
        self.events.add("S2_IGNITION", self.t_s2_ignition,
                        "LE-5B-3 ignition")
        self.events.add("SECO", self.t_seco,
                        "Second Engine Cut-Off — orbit insertion")

    def _initial_state(self) -> FlightState:
        """打上げ初期状態"""
        state = FlightState()
        state.t = 0.0
        state.z = 0.0
        state.vz = 0.01  # 微小初速 (数値安定性)
        state.pitch = math.radians(90)
        state.gamma = math.radians(90)
        state.mass = self.propulsion.liftoff_mass
        state.phase = "P1_SRB_S1"
        state.update_derived()
        return state

    def _handle_events(self, fired: List[str], state: FlightState):
        """イベント発火時の処理"""
        for event_name in fired:
            if event_name == "SRB_SEP":
                # SRB 分離: ケース質量投棄 + 姿勢外乱
                self._srb_active = False
                srb_mass = sum(b.dry_mass for b in self.propulsion.srb_boosters)
                state.mass -= srb_mass
                state.pitch_rate += math.radians(self.srb_sep_impulse_pitch)
                state.phase = "P2_S1_ONLY"

            elif event_name == "FAIRING_SEP":
                self._fairing_on = False
                state.mass -= self.propulsion.fairing_mass
                state.phase = "P3_S1_COAST"

            elif event_name == "MECO":
                self._s1_active = False
                state.phase = "COAST"

            elif event_name == "STAGE_SEP":
                # 段間分離: 2段+ペイロード質量に設定 (1段構造を投棄)
                s2_mass = (self.propulsion.s2_propellant
                           + self.propulsion.s2_dry
                           + self.propulsion.payload_mass)
                state.mass = s2_mass
                state.pitch_rate += math.radians(self.stage_sep_impulse_pitch)
                # ピッチ角をフライト経路角に再同期 (分離外乱リセット)
                state.pitch = state.gamma
                state.pitch_rate = 0.0
                state.phase = "STAGE_SEP"

            elif event_name == "S2_IGNITION":
                self._s2_active = True
                state.phase = "P4_S2_BURN"

            elif event_name == "SECO":
                self._s2_active = False
                state.phase = "ORBIT"

    def run_mission(self, t_end: float = None) -> dict:
        """
        ミッションシミュレーション実行

        Args:
            t_end: シミュレーション終了時刻 [s] (default: SECO + 10s)

        Returns:
            dict: テレメトリデータ (numpy arrays) + events log
        """
        if t_end is None:
            t_end = self.t_seco + 10.0

        state = self._initial_state()
        self.telemetry = TelemetryRecorder()
        self.attitude.reset()
        self._srb_active = self.propulsion.n_srb > 0
        self._s1_active = True
        self._s2_active = False
        self._fairing_on = True

        # 推進剤残量追跡
        self._s1_prop_remaining = self.propulsion.s1_propellant
        self._s2_prop_remaining = self.propulsion.s2_propellant
        self._srb_prop_remaining = sum(
            b.propellant_mass for b in self.propulsion.srb_boosters)

        # 温度追跡
        T_nose = 288.0
        T_body = 288.0
        C_nose = (self.thermal.tps_nose.rho * self.thermal.tps_nose.cp
                  * self.thermal.tps_nose.thickness)
        C_body = (self.thermal.tps_body.rho * self.thermal.tps_body.cp
                  * self.thermal.tps_body.thickness)
        eps_n = self.thermal.tps_nose.emissivity
        eps_b = self.thermal.tps_body.emissivity
        sigma = 5.67e-8

        n_steps = int(t_end / self.dt) + 1
        dt = self.dt

        for step in range(n_steps):
            t = step * dt
            state.t = t

            # ── 1. イベント判定 ──
            fired = self.events.check(t, state)
            if fired:
                self._handle_events(fired, state)

            # ── 2. 推進力 ──
            # 推進剤枯渇チェック
            s1_active = self._s1_active and self._s1_prop_remaining > 0
            s2_active = self._s2_active and self._s2_prop_remaining > 0

            prop = self.propulsion.get_total_thrust(
                t=t, altitude=state.z,
                s1_throttle=1.0,
                srb_active=self._srb_active,
                s1_active=s1_active,
                s2_active=s2_active,
            )
            thrust = prop["thrust"]
            mdot = prop["mdot"]

            # 推進剤消費内訳
            # LE-9 → S1 液体推進剤, LE-5B → S2 液体推進剤
            # SRB → 自前の固体推進剤 (質量は SRB 構造に含む)
            mdot_s1 = 0.0
            mdot_s2 = 0.0
            mdot_srb = 0.0
            for key, comp in prop["components"].items():
                if key.startswith("LE-9"):
                    mdot_s1 += comp["mdot"]
                elif key.startswith("LE-5B"):
                    mdot_s2 += comp["mdot"]
                elif key.startswith("SRB"):
                    mdot_srb += comp["mdot"]

            # ── 3. 空力 ──
            alpha_deg = math.degrees(state.alpha)
            alpha_deg = np.clip(alpha_deg, -10, 10)
            if state.z < 150_000 and state.speed > 1.0:
                aero_result = self.aero.get_forces(state.mach, alpha_deg, state.z)
                drag = aero_result["drag"]
                lift = aero_result["lift"]
                normal = aero_result["normal"]
            else:
                drag = 0.0
                lift = 0.0
                normal = 0.0
                aero_result = {"Cd": 0, "q_dyn": 0}

            # Cd 変化: フェアリング分離後
            if not self._fairing_on:
                drag *= 0.7  # フェアリング無しで抗力減少

            # ── 4. 姿勢制御 ──
            att_state = AttitudeState(
                pitch=state.pitch, yaw=state.yaw, roll=state.roll,
                pitch_rate=state.pitch_rate, yaw_rate=state.yaw_rate,
                roll_rate=state.roll_rate,
                time=t, altitude=state.z, mach=state.mach,
                alpha=state.alpha, q_dyn=state.q_dyn,
            )
            cmd = self.attitude.update(t, att_state, dt)
            gimbal_pitch = cmd["gimbal_pitch"]

            # ── 5. 運動方程式 (3DOF) ──
            # 推力方向 (ピッチ角 + ジンバル偏向)
            thrust_angle = state.pitch + math.radians(gimbal_pitch)

            # 力の合算
            Fx = (thrust * math.cos(thrust_angle)
                  - drag * math.cos(state.gamma))
            Fz = (thrust * math.sin(thrust_angle)
                  - drag * math.sin(state.gamma)
                  - state.mass * state.g_local
                  + lift)

            # 加速度
            ax = Fx / state.mass
            az = Fz / state.mass

            # 速度・位置更新 (Euler)
            state.vx += ax * dt
            state.vz += az * dt
            state.x += state.vx * dt
            state.z += state.vz * dt
            state.z = max(state.z, 0.0)

            # 推進剤消費・質量更新
            dm_s1 = min(mdot_s1 * dt, self._s1_prop_remaining)
            dm_s2 = min(mdot_s2 * dt, self._s2_prop_remaining)
            dm_srb = min(mdot_srb * dt, self._srb_prop_remaining)
            self._s1_prop_remaining -= dm_s1
            self._s2_prop_remaining -= dm_s2
            self._srb_prop_remaining -= dm_srb
            state.mass -= (dm_s1 + dm_s2 + dm_srb)
            state.mass = max(state.mass, self.propulsion.payload_mass)

            # 加速度記録
            accel_total = math.sqrt(ax**2 + az**2)
            state.accel = accel_total
            state.accel_axial = thrust / state.mass / G0 if state.mass > 0 else 0

            # 姿勢ダイナミクス (簡易: 制御モーメントによる更新)
            mass_ratio = max(0.1, state.mass / self.propulsion.liftoff_mass)
            Iyy = self.attitude.Iyy * mass_ratio
            if abs(cmd["moment_pitch"]) > 0:
                alpha_pitch = cmd["moment_pitch"] / Iyy
                state.pitch_rate += alpha_pitch * dt
            # ピッチレートダンピング (数値安定性)
            state.pitch_rate *= 0.999
            state.pitch += state.pitch_rate * dt
            # ピッチ角の範囲制限 (-90° ~ +180°)
            state.pitch = max(math.radians(-90), min(math.radians(180), state.pitch))

            # 導出パラメータ更新
            state.update_derived()

            # ── 6. 空力加熱 ──
            if state.z < 150_000 and state.speed > 10:
                atm = atmosphere_isa(state.z)
                q_stag = self.thermal.stagnation_heating(atm["rho"], state.speed)
                q_body = q_stag * 0.08

                # ノーズ温度更新
                q_rad_n = eps_n * sigma * T_nose**4
                T_nose += (q_stag - q_rad_n) / C_nose * dt
                T_nose = max(T_nose, atm["T"])

                # ボディ温度更新
                q_rad_b = eps_b * sigma * T_body**4
                T_body += (q_body - q_rad_b) / C_body * dt
                T_body = max(T_body, atm["T"])
            else:
                q_stag = 0.0
                # 放射冷却
                if T_nose > 200:
                    T_nose -= eps_n * sigma * T_nose**4 / C_nose * dt
                    T_nose = max(T_nose, 3.0)  # CMB temperature floor
                if T_body > 200:
                    T_body -= eps_b * sigma * T_body**4 / C_body * dt
                    T_body = max(T_body, 3.0)

            # ── 7. テレメトリ記録 ──
            extras = {
                "thrust": thrust,
                "drag": drag,
                "accel_G": state.accel / G0,
                "accel_axial_G": state.accel_axial,
                "gimbal_pitch": gimbal_pitch,
                "T_nose": T_nose,
                "T_body": T_body,
                "q_stag": q_stag,
            }
            self.telemetry.record(state, extras)

            # 落下チェック
            if state.z < -100 and t > 10:
                break

        result = self.telemetry.to_arrays()
        result["events"] = self.events.log
        return result

    def mission_summary(self, result: dict) -> str:
        """ミッション結果サマリー"""
        t = result["t"]
        alt = result["altitude"]
        speed = result["speed"]
        mach = result["mach"]
        mass = result["mass"]
        q_dyn = result["q_dyn"]
        thrust = result["thrust"]
        accel_G = result["accel_G"]
        T_nose = result["T_nose"]

        # Max-Q
        i_maxq = np.argmax(q_dyn)

        # 最大加速度
        i_max_accel = np.argmax(accel_G)

        # SECO 時の状態
        i_end = len(t) - 1

        lines = [
            f"{'='*60}",
            f"  H3 Virtual Twin — Mission Summary ({self.config})",
            f"{'='*60}",
            "",
            f"  Liftoff mass:      {mass[0]/1e3:.1f} ton",
            f"  Final mass:        {mass[i_end]/1e3:.1f} ton",
            f"  Mission duration:  {t[i_end]:.1f} s",
            "",
            "  --- Key Events ---",
        ]
        for evt_t, evt_name, evt_desc in result["events"]:
            idx = min(int(evt_t / self.dt), i_end)
            lines.append(
                f"    T+{evt_t:6.1f}s  {evt_name:15s}  "
                f"h={alt[idx]/1e3:.1f}km  V={speed[idx]:.0f}m/s  "
                f"M={mach[idx]:.1f}"
            )

        lines.extend([
            "",
            "  --- Performance ---",
            f"  Max-Q:           {q_dyn[i_maxq]/1e3:.1f} kPa "
            f"at T+{t[i_maxq]:.0f}s (h={alt[i_maxq]/1e3:.1f}km, M={mach[i_maxq]:.1f})",
            f"  Max accel:       {accel_G[i_max_accel]:.1f} G "
            f"at T+{t[i_max_accel]:.0f}s",
            f"  Max nose temp:   {np.max(T_nose):.0f} K",
            "",
            "  --- Final State (SECO) ---",
            f"  Altitude:        {alt[i_end]/1e3:.1f} km",
            f"  Velocity:        {speed[i_end]:.0f} m/s",
            f"  Downrange:       {result['x'][i_end]/1e3:.1f} km",
            f"  Flight path:     {result['gamma_deg'][i_end]:.1f}°",
            f"  Mass:            {mass[i_end]/1e3:.1f} ton",
        ])

        # 軌道パラメータ概算
        r = RE + alt[i_end]
        v = speed[i_end]
        E = 0.5 * v**2 - MU / r
        a_orbit = -MU / (2 * E) if E < 0 else float('inf')
        if a_orbit > 0 and a_orbit < 1e9:
            v_circ = math.sqrt(MU / r)
            lines.extend([
                "",
                "  --- Orbit Estimate ---",
                f"  Semi-major axis:  {a_orbit/1e3:.0f} km",
                f"  Orbital altitude: {(a_orbit - RE)/1e3:.0f} km (circular equiv.)",
                f"  V_circular:       {v_circ:.0f} m/s",
                f"  Delta-V deficit:  {max(0, v_circ - v):.0f} m/s",
            ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 実行テスト
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("H3 Virtual Twin — Flight Orchestrator Test")
    print("=" * 60)

    # H3-22S ミッション
    orch = H3FlightOrchestrator("H3-22S", dt=0.5)
    print(f"\nRunning H3-22S mission simulation...")
    result = orch.run_mission()
    print(orch.mission_summary(result))

    # 飛行プロファイル (抜粋)
    print("\n  --- Flight Profile ---")
    print(f"  {'T[s]':>6s}  {'h[km]':>7s}  {'V[m/s]':>7s}  {'M':>5s}  "
          f"{'mass[t]':>7s}  {'q[kPa]':>7s}  {'F[kN]':>7s}  "
          f"{'a[G]':>5s}  {'θ[°]':>6s}  {'T_n[K]':>6s}  {'phase'}")
    for t_snap in [0, 10, 30, 60, 90, 113, 116, 120, 150, 200,
                   230, 250, 290, 298, 303, 308, 400, 500, 700, 850, 860]:
        idx = int(t_snap / orch.dt)
        if idx >= len(result["t"]):
            continue
        print(f"  {result['t'][idx]:6.1f}  "
              f"{result['altitude'][idx]/1e3:7.1f}  "
              f"{result['speed'][idx]:7.0f}  "
              f"{result['mach'][idx]:5.1f}  "
              f"{result['mass'][idx]/1e3:7.1f}  "
              f"{result['q_dyn'][idx]/1e3:7.1f}  "
              f"{result['thrust'][idx]/1e3:7.0f}  "
              f"{result['accel_G'][idx]:5.1f}  "
              f"{result['pitch_deg'][idx]:6.1f}  "
              f"{result['T_nose'][idx]:6.0f}  "
              f"{result['phase'][idx]}")

    # H3-30S (SRBなし)
    print(f"\n{'='*60}")
    print("Running H3-30S mission simulation...")
    orch30 = H3FlightOrchestrator("H3-30S", dt=0.5)
    result30 = orch30.run_mission()
    print(orch30.mission_summary(result30))
