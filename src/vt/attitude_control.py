#!/usr/bin/env python3
"""
H3 Virtual Twin — Attitude Control Module (A6)

ピッチ/ヨー/ロール 3軸 PID 姿勢制御 + TVC (Thrust Vector Control)。
LE-9 ジンバル角制限・SRB 推力偏向を考慮。

制御構成:
  1段: LE-9 ジンバル (±7°) + SRB TVC (±5°)
  2段: LE-5B-3 ジンバル (±3°) + RCS (ロール)
  フェアリング分離後: ロール RCS のみ

References:
  - JAXA H3 Rocket User's Manual
  - Zipfel, "Modeling and Simulation of Aerospace Vehicle Dynamics"
  - gncpy (attitude control library)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── 定数 ──
DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi


# ═══════════════════════════════════════════════════════════════
# PID コントローラ
# ═══════════════════════════════════════════════════════════════

@dataclass
class PIDController:
    """
    PID コントローラ (anti-windup 付き)

    離散時間実装: backward Euler 積分 + derivative filter
    """
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    # Anti-windup
    integral_limit: float = 10.0
    # Output limit
    output_min: float = -1.0
    output_max: float = 1.0
    # Derivative filter (1st order, tau [s])
    d_filter_tau: float = 0.05

    def __post_init__(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        PID 更新

        Args:
            error: 偏差 (目標 - 実値)
            dt: 時間ステップ [s]

        Returns:
            制御出力 (output_min ~ output_max)
        """
        if dt <= 0:
            return 0.0

        # P
        P = self.Kp * error

        # I (anti-windup: clamp)
        self._integral += error * dt
        self._integral = np.clip(self._integral,
                                 -self.integral_limit, self.integral_limit)
        I = self.Ki * self._integral

        # D (filtered)
        raw_d = (error - self._prev_error) / dt
        alpha = dt / (self.d_filter_tau + dt)
        filtered_d = alpha * raw_d + (1 - alpha) * self._prev_derivative
        D = self.Kd * filtered_d
        self._prev_derivative = filtered_d
        self._prev_error = error

        output = P + I + D
        return float(np.clip(output, self.output_min, self.output_max))


# ═══════════════════════════════════════════════════════════════
# 飛行状態
# ═══════════════════════════════════════════════════════════════

@dataclass
class AttitudeState:
    """姿勢状態ベクトル"""
    # オイラー角 [rad]
    pitch: float = math.radians(90)   # θ (打上げ時: 90° = 鉛直)
    yaw: float = 0.0                  # ψ
    roll: float = 0.0                 # φ
    # 角速度 [rad/s]
    pitch_rate: float = 0.0           # q
    yaw_rate: float = 0.0             # r
    roll_rate: float = 0.0            # p
    # 飛行情報
    time: float = 0.0                 # [s]
    altitude: float = 0.0             # [m]
    mach: float = 0.0
    alpha: float = 0.0               # 迎角 [rad]
    beta: float = 0.0                # 横滑り角 [rad]
    q_dyn: float = 0.0               # 動圧 [Pa]


# ═══════════════════════════════════════════════════════════════
# TVC (Thrust Vector Control)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TVCActuator:
    """TVC ジンバルアクチュエータ"""
    name: str = "LE-9 TVC"
    max_deflection: float = 7.0      # 最大偏向角 [deg]
    rate_limit: float = 20.0         # 偏向速度制限 [deg/s]
    bandwidth: float = 8.0           # 帯域幅 [Hz]
    # 現在値
    _deflection_pitch: float = 0.0   # [deg]
    _deflection_yaw: float = 0.0     # [deg]

    def command(self, pitch_cmd: float, yaw_cmd: float, dt: float) -> tuple:
        """
        ジンバル角指令 → レート制限付きで適用

        Args:
            pitch_cmd: ピッチ偏向指令 [deg]
            yaw_cmd: ヨー偏向指令 [deg]
            dt: 時間ステップ [s]

        Returns:
            (actual_pitch_deg, actual_yaw_deg)
        """
        max_d = self.max_deflection

        # レート制限
        max_delta = self.rate_limit * dt

        # ピッチ
        target_p = np.clip(pitch_cmd, -max_d, max_d)
        delta_p = target_p - self._deflection_pitch
        delta_p = np.clip(delta_p, -max_delta, max_delta)
        self._deflection_pitch += delta_p

        # ヨー
        target_y = np.clip(yaw_cmd, -max_d, max_d)
        delta_y = target_y - self._deflection_yaw
        delta_y = np.clip(delta_y, -max_delta, max_delta)
        self._deflection_yaw += delta_y

        return (self._deflection_pitch, self._deflection_yaw)

    def reset(self):
        self._deflection_pitch = 0.0
        self._deflection_yaw = 0.0


# ═══════════════════════════════════════════════════════════════
# Gravity Turn プロファイル
# ═══════════════════════════════════════════════════════════════

@dataclass
class GravityTurnProfile:
    """
    重力ターンプログラム角度 (目標ピッチ角 vs 時刻)

    H3 は打上げ直後にピッチオーバーし、重力ターンに移行。
    """
    # ピッチオーバー開始
    t_pitchover_start: float = 10.0     # [s]
    pitchover_rate: float = -0.8        # [deg/s] (負 = nose down)
    pitchover_duration: float = 8.0     # [s]

    # 重力ターン遷移
    t_gravity_turn: float = 30.0        # [s] 重力ターン開始
    # 以降は速度ベクトルに追従

    # ステージイベント
    t_srb_sep: float = 113.0            # SRB 分離
    t_fairing_sep: float = 230.0        # フェアリング分離
    t_meco: float = 290.0               # 1段エンジン停止
    t_s2_ignition: float = 300.0        # 2段点火
    t_seco: float = 900.0               # 2段エンジン停止

    def target_pitch(self, t: float, altitude: float = 0,
                     velocity: float = 0) -> float:
        """
        目標ピッチ角 [deg] (慣性座標)

        90° = 鉛直, 0° = 水平
        """
        if t < self.t_pitchover_start:
            return 90.0
        elif t < self.t_pitchover_start + self.pitchover_duration:
            dt = t - self.t_pitchover_start
            return 90.0 + self.pitchover_rate * dt
        elif t < self.t_gravity_turn:
            # ピッチオーバー後 → 重力ターン前の遷移
            pitch_after_po = 90.0 + self.pitchover_rate * self.pitchover_duration
            progress = (t - self.t_pitchover_start - self.pitchover_duration) / (
                self.t_gravity_turn - self.t_pitchover_start - self.pitchover_duration)
            # 重力ターン目標へスムーズ遷移
            gt_target = self._gravity_turn_angle(t, altitude, velocity)
            return pitch_after_po + (gt_target - pitch_after_po) * progress
        else:
            return self._gravity_turn_angle(t, altitude, velocity)

    def _gravity_turn_angle(self, t: float, altitude: float,
                            velocity: float) -> float:
        """重力ターン中のピッチ角 (簡易: 時間ベース減衰)"""
        # 実際は速度ベクトル追従だが、ここでは時間ベースの参照軌道
        if t < self.t_meco:
            # 1段: 90° → ~15° (線形減衰 + 曲線)
            progress = (t - self.t_gravity_turn) / (self.t_meco - self.t_gravity_turn)
            return 83.0 * (1 - progress)**1.3 + 5.0
        elif t < self.t_s2_ignition:
            # コースト
            return 8.0
        elif t < self.t_seco:
            # 2段: 8° → 0° (軌道投入)
            progress = (t - self.t_s2_ignition) / (self.t_seco - self.t_s2_ignition)
            return 8.0 * (1 - progress)
        else:
            return 0.0

    def target_yaw(self, t: float) -> float:
        """目標ヨー角 [deg] (射場方位角に依存、ここでは 0 固定)"""
        return 0.0

    def target_roll(self, t: float) -> float:
        """目標ロール角 [deg]"""
        return 0.0


# ═══════════════════════════════════════════════════════════════
# H3 姿勢制御系
# ═══════════════════════════════════════════════════════════════

@dataclass
class H3AttitudeController:
    """
    H3 ロケット 3軸姿勢制御系

    制御ループ:
      1. 目標姿勢 (GravityTurnProfile)
      2. 偏差計算
      3. PID → ジンバル角指令
      4. TVC アクチュエータ → 実偏向角
      5. モーメント計算 → 角加速度

    Flight Orchestrator 向けインターフェース:
      cmd = controller.update(t, state)
    """
    config: str = "H3-22S"
    profile: GravityTurnProfile = field(default_factory=GravityTurnProfile)

    # PID ゲイン (ピッチ)
    pid_pitch: PIDController = field(default_factory=lambda: PIDController(
        Kp=2.5, Ki=0.3, Kd=1.2,
        integral_limit=5.0,
        output_min=-7.0, output_max=7.0,  # [deg]
        d_filter_tau=0.02,
    ))
    # PID ゲイン (ヨー)
    pid_yaw: PIDController = field(default_factory=lambda: PIDController(
        Kp=2.5, Ki=0.3, Kd=1.2,
        integral_limit=5.0,
        output_min=-7.0, output_max=7.0,
        d_filter_tau=0.02,
    ))
    # PID ゲイン (ロール) — RCS
    pid_roll: PIDController = field(default_factory=lambda: PIDController(
        Kp=3.0, Ki=0.5, Kd=0.8,
        integral_limit=3.0,
        output_min=-1.0, output_max=1.0,  # normalized
        d_filter_tau=0.02,
    ))

    # TVC
    tvc_le9: TVCActuator = field(default_factory=lambda: TVCActuator(
        "LE-9 TVC", max_deflection=7.0, rate_limit=20.0))
    tvc_le5b: TVCActuator = field(default_factory=lambda: TVCActuator(
        "LE-5B-3 TVC", max_deflection=3.0, rate_limit=15.0))

    # 機体慣性モーメント (打上げ時概算)
    Iyy: float = 8.0e7     # ピッチ [kg·m²]
    Izz: float = 8.0e7     # ヨー [kg·m²]
    Ixx: float = 2.0e6     # ロール [kg·m²]

    # LE-9 ジンバル点と重心の距離
    gimbal_arm: float = 30.0   # [m] (重心からエンジンまでの距離)

    def __post_init__(self):
        self._phase = "s1"  # "s1", "coast", "s2", "done"

    def reset(self):
        self.pid_pitch.reset()
        self.pid_yaw.reset()
        self.pid_roll.reset()
        self.tvc_le9.reset()
        self.tvc_le5b.reset()
        self._phase = "s1"

    def update(self, t: float, state: AttitudeState, dt: float = 0.02) -> dict:
        """
        姿勢制御 1ステップ

        Args:
            t: 時刻 [s]
            state: 現在の姿勢状態
            dt: 制御周期 [s]

        Returns:
            dict: gimbal_pitch[deg], gimbal_yaw[deg], roll_cmd[-1..1],
                  target_pitch[deg], pitch_error[deg],
                  moment_pitch[Nm], moment_yaw[Nm], moment_roll[Nm]
        """
        # フェーズ更新
        if t >= self.profile.t_seco:
            self._phase = "done"
        elif t >= self.profile.t_s2_ignition:
            self._phase = "s2"
        elif t >= self.profile.t_meco:
            self._phase = "coast"
        else:
            self._phase = "s1"

        # 目標姿勢
        target_pitch = self.profile.target_pitch(t, state.altitude)
        target_yaw = self.profile.target_yaw(t)
        target_roll = self.profile.target_roll(t)

        # 偏差 [deg]
        pitch_error = target_pitch - math.degrees(state.pitch)
        yaw_error = target_yaw - math.degrees(state.yaw)
        roll_error = target_roll - math.degrees(state.roll)

        # 角速度フィードバック (rate damping)
        pitch_rate_deg = math.degrees(state.pitch_rate)
        yaw_rate_deg = math.degrees(state.yaw_rate)
        roll_rate_deg = math.degrees(state.roll_rate)

        # PID
        gimbal_pitch_cmd = self.pid_pitch.update(pitch_error - 0.5 * pitch_rate_deg, dt)
        gimbal_yaw_cmd = self.pid_yaw.update(yaw_error - 0.5 * yaw_rate_deg, dt)
        roll_cmd = self.pid_roll.update(roll_error - 0.3 * roll_rate_deg, dt)

        # TVC アクチュエータ
        if self._phase == "s1":
            gp, gy = self.tvc_le9.command(gimbal_pitch_cmd, gimbal_yaw_cmd, dt)
        elif self._phase == "s2":
            gp, gy = self.tvc_le5b.command(gimbal_pitch_cmd, gimbal_yaw_cmd, dt)
        else:
            gp, gy = 0.0, 0.0

        # 制御モーメント計算
        # M = F * L * sin(δ) ≈ F * L * δ [rad] (小角近似)
        if self._phase == "s1":
            F_engine = 1_471_000.0 * 2  # LE-9 x2
            arm = self.gimbal_arm
        elif self._phase == "s2":
            F_engine = 137_200.0
            arm = 8.0  # 2段は短い
        else:
            F_engine = 0.0
            arm = 0.0

        M_pitch = F_engine * arm * math.sin(math.radians(gp))
        M_yaw = F_engine * arm * math.sin(math.radians(gy))

        # RCS ロールモーメント
        rcs_torque = 5000.0  # [Nm] RCS 最大トルク
        M_roll = roll_cmd * rcs_torque

        # 風外乱モーメント (簡易)
        M_wind_pitch = 0.0
        M_wind_yaw = 0.0
        if state.q_dyn > 0 and abs(state.alpha) > 0.001:
            # 空力モーメント ≈ q * S * d * Cn * (Xcp - Xcg)
            S_ref = 21.24  # [m²]
            Cn_alpha = 2.0  # [/rad]
            cp_cg_offset = -3.0  # Xcp - Xcg [m] (安定なら負)
            M_wind_pitch = state.q_dyn * S_ref * Cn_alpha * state.alpha * cp_cg_offset

        return {
            "gimbal_pitch": gp,
            "gimbal_yaw": gy,
            "roll_cmd": roll_cmd,
            "target_pitch": target_pitch,
            "pitch_error": pitch_error,
            "yaw_error": yaw_error,
            "moment_pitch": M_pitch + M_wind_pitch,
            "moment_yaw": M_yaw + M_wind_yaw,
            "moment_roll": M_roll,
            "phase": self._phase,
        }

    def simulate(self, t_end: float = 300.0, dt: float = 0.02,
                 wind_profile: Optional[callable] = None) -> dict:
        """
        姿勢制御シミュレーション (3-DOF rotational)

        簡易運動方程式:
          I * dω/dt = M_control + M_aero + M_gravity_gradient
          dθ/dt = ω

        Args:
            t_end: シミュレーション終了時刻 [s]
            dt: 時間ステップ [s]
            wind_profile: 風外乱関数 wind(t, alt) → (Fx, Fy) [N]

        Returns:
            dict: time, pitch, yaw, roll, pitch_rate, yaw_rate, roll_rate,
                  gimbal_pitch, gimbal_yaw, target_pitch, altitude, mach arrays
        """
        self.reset()

        n = int(t_end / dt) + 1
        time = np.linspace(0, t_end, n)

        # 出力配列
        pitch = np.zeros(n)
        yaw = np.zeros(n)
        roll = np.zeros(n)
        p_rate = np.zeros(n)
        y_rate = np.zeros(n)
        r_rate = np.zeros(n)
        gp_arr = np.zeros(n)
        gy_arr = np.zeros(n)
        tgt_pitch = np.zeros(n)
        alt_arr = np.zeros(n)
        mach_arr = np.zeros(n)
        err_arr = np.zeros(n)

        # 初期状態
        state = AttitudeState()
        state.pitch = math.radians(90)  # 鉛直

        # 簡易軌道 (altitude, velocity)
        for i in range(n):
            t = time[i]

            # 簡易軌道モデル
            if t < 290:  # 1段燃焼中
                alt = 350 * t + 0.5 * 15 * t**2
                alt = min(alt, 200_000)
                vel = 30 + 22 * t
            elif t < 300:
                alt = 200_000
                vel = 30 + 22 * 290
            else:
                alt = 200_000 + 100 * (t - 300)
                vel = 30 + 22 * 290 + 10 * (t - 300)
            vel = min(vel, 7800)

            a_sound = 340 if alt < 11000 else 295
            mach = vel / a_sound
            q_dyn = 0.5 * 1.225 * math.exp(-alt / 8500) * vel**2 if alt < 100000 else 0.0

            state.time = t
            state.altitude = alt
            state.mach = mach
            state.q_dyn = q_dyn
            state.alpha = math.radians(max(0, math.degrees(state.pitch) -
                          self.profile.target_pitch(t, alt)))

            # 風外乱
            if wind_profile is not None:
                fx, fy = wind_profile(t, alt)
                state.alpha += fy / max(q_dyn * 21.24, 1.0)

            # 制御
            cmd = self.update(t, state, dt)

            # 記録
            pitch[i] = math.degrees(state.pitch)
            yaw[i] = math.degrees(state.yaw)
            roll[i] = math.degrees(state.roll)
            p_rate[i] = math.degrees(state.pitch_rate)
            y_rate[i] = math.degrees(state.yaw_rate)
            r_rate[i] = math.degrees(state.roll_rate)
            gp_arr[i] = cmd["gimbal_pitch"]
            gy_arr[i] = cmd["gimbal_yaw"]
            tgt_pitch[i] = cmd["target_pitch"]
            alt_arr[i] = alt
            mach_arr[i] = mach
            err_arr[i] = cmd["pitch_error"]

            # 慣性モーメント更新 (質量減少に伴う)
            mass_ratio = max(0.3, 1.0 - t / 600)
            Iyy_t = self.Iyy * mass_ratio
            Izz_t = self.Izz * mass_ratio
            Ixx_t = self.Ixx * mass_ratio

            # 角加速度
            alpha_pitch = cmd["moment_pitch"] / Iyy_t
            alpha_yaw = cmd["moment_yaw"] / Izz_t
            alpha_roll = cmd["moment_roll"] / Ixx_t

            # 積分
            state.pitch_rate += alpha_pitch * dt
            state.yaw_rate += alpha_yaw * dt
            state.roll_rate += alpha_roll * dt

            state.pitch += state.pitch_rate * dt
            state.yaw += state.yaw_rate * dt
            state.roll += state.roll_rate * dt

        return {
            "time": time,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "pitch_rate": p_rate,
            "yaw_rate": y_rate,
            "roll_rate": r_rate,
            "gimbal_pitch": gp_arr,
            "gimbal_yaw": gy_arr,
            "target_pitch": tgt_pitch,
            "pitch_error": err_arr,
            "altitude": alt_arr,
            "mach": mach_arr,
        }

    def summary(self) -> str:
        lines = [
            f"=== H3 Attitude Controller ({self.config}) ===",
            f"  Phase: {self._phase}",
            f"  Pitch PID: Kp={self.pid_pitch.Kp}, Ki={self.pid_pitch.Ki}, "
            f"Kd={self.pid_pitch.Kd}",
            f"  Yaw PID:   Kp={self.pid_yaw.Kp}, Ki={self.pid_yaw.Ki}, "
            f"Kd={self.pid_yaw.Kd}",
            f"  Roll PID:  Kp={self.pid_roll.Kp}, Ki={self.pid_roll.Ki}, "
            f"Kd={self.pid_roll.Kd}",
            f"  LE-9 TVC:  ±{self.tvc_le9.max_deflection}° "
            f"({self.tvc_le9.rate_limit}°/s)",
            f"  LE-5B TVC: ±{self.tvc_le5b.max_deflection}° "
            f"({self.tvc_le5b.rate_limit}°/s)",
            f"  Iyy: {self.Iyy:.1e} kg·m²",
            f"  Gimbal arm: {self.gimbal_arm:.0f} m",
            "",
            "  Gravity Turn Profile:",
            f"    Pitchover: t={self.profile.t_pitchover_start:.0f}s, "
            f"rate={self.profile.pitchover_rate:.1f}°/s, "
            f"duration={self.profile.pitchover_duration:.0f}s",
            f"    Gravity turn: t={self.profile.t_gravity_turn:.0f}s",
            f"    SRB sep: t={self.profile.t_srb_sep:.0f}s",
            f"    Fairing sep: t={self.profile.t_fairing_sep:.0f}s",
            f"    MECO: t={self.profile.t_meco:.0f}s",
            f"    S2 ignition: t={self.profile.t_s2_ignition:.0f}s",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 実行テスト
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("H3 Virtual Twin — Attitude Control Module Test")
    print("=" * 60)

    ctrl = H3AttitudeController("H3-22S")
    print(ctrl.summary())

    # Gravity Turn プロファイル
    print("\n--- Gravity Turn Target Pitch ---")
    for t in [0, 5, 10, 14, 18, 25, 30, 60, 90, 120, 150,
              200, 250, 290, 300, 500, 900]:
        p = ctrl.profile.target_pitch(t)
        print(f"  t={t:4d}s: θ_target={p:6.1f}°")

    # 制御シミュレーション
    print("\n--- Attitude Control Simulation (300s) ---")
    result = ctrl.simulate(t_end=300.0, dt=0.02)

    print(f"\n  Time snapshots:")
    print(f"  {'t[s]':>5s}  {'θ_tgt[°]':>8s}  {'θ_act[°]':>8s}  "
          f"{'err[°]':>7s}  {'δ_gim[°]':>8s}  {'q[°/s]':>7s}  "
          f"{'h[km]':>6s}  {'M':>5s}")

    for t_snap in [0, 10, 15, 20, 30, 60, 90, 120, 150, 200, 250, 290, 300]:
        idx = int(t_snap / 0.02)
        if idx >= len(result["time"]):
            continue
        print(f"  {result['time'][idx]:5.0f}  "
              f"{result['target_pitch'][idx]:8.1f}  "
              f"{result['pitch'][idx]:8.1f}  "
              f"{result['pitch_error'][idx]:7.2f}  "
              f"{result['gimbal_pitch'][idx]:8.2f}  "
              f"{result['pitch_rate'][idx]:7.2f}  "
              f"{result['altitude'][idx]/1000:6.1f}  "
              f"{result['mach'][idx]:5.1f}")

    # 制御性能指標
    err = result["pitch_error"]
    print(f"\n  Performance metrics:")
    print(f"    Max pitch error:   {np.max(np.abs(err)):.2f}°")
    print(f"    RMS pitch error:   {np.sqrt(np.mean(err**2)):.2f}°")
    print(f"    Max gimbal angle:  {np.max(np.abs(result['gimbal_pitch'])):.2f}°")
    print(f"    Max pitch rate:    {np.max(np.abs(result['pitch_rate'])):.2f}°/s")

    # 風外乱テスト
    print("\n--- With Wind Disturbance ---")
    def wind(t, alt):
        # ジェット気流 (10-15km で最大)
        w = 50 * math.exp(-((alt - 12000) / 5000)**2)  # [m/s]
        Fy = 0.5 * 1.225 * math.exp(-alt / 8500) * w**2 * 21.24 * 0.01
        return (0, Fy)

    result_wind = ctrl.simulate(t_end=300.0, dt=0.02, wind_profile=wind)
    err_w = result_wind["pitch_error"]
    print(f"    Max pitch error:   {np.max(np.abs(err_w)):.2f}°")
    print(f"    RMS pitch error:   {np.sqrt(np.mean(err_w**2)):.2f}°")
    print(f"    Max gimbal angle:  {np.max(np.abs(result_wind['gimbal_pitch'])):.2f}°")
