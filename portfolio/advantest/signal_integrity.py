#!/usr/bin/env python3
"""
Signal Integrity (SI) / Power Integrity (PI) Analysis Tool
============================================================
Portfolio: Advantest Corporation — Semiconductor Test Equipment

Transmission line analysis using Telegrapher's equations for
high-speed digital interconnects in ATE (Automatic Test Equipment).

Theory:
- Lossy transmission line: RLGC model
- Telegrapher's equations → characteristic impedance Z0(f), propagation γ(f)
- S-parameters from ABCD matrix
- Eye diagram via inverse FFT
- TDR (Time-Domain Reflectometry) for impedance profiling
- PDN (Power Distribution Network) impedance

Author: Nishioka
"""

import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import time

# ============================================================
# Physical constants
# ============================================================
c0 = 2.99792458e8
eps0 = 8.854187817e-12
mu0 = 4e-7 * np.pi

# ============================================================
# Transmission Line RLGC Model
# ============================================================
class TransmissionLine:
    """
    Lossy transmission line with frequency-dependent RLGC parameters.

    Telegrapher's equations:
    dV/dz = -(R + jωL) I
    dI/dz = -(G + jωC) V

    Propagation constant: γ = α + jβ = sqrt((R+jωL)(G+jωC))
    Characteristic impedance: Z0 = sqrt((R+jωL)/(G+jωC))
    """

    def __init__(self, R0=2.0, L0=250e-9, G0=0.0, C0=100e-12,
                 f_skin=1e9, length=0.1):
        """
        R0: DC resistance [Ω/m]
        L0: Inductance per unit length [H/m]
        G0: DC conductance [S/m]
        C0: Capacitance per unit length [F/m]
        f_skin: Skin-effect onset frequency [Hz]
        length: Line length [m]
        """
        self.R0 = R0
        self.L0 = L0
        self.G0 = G0
        self.C0 = C0
        self.f_skin = f_skin
        self.length = length

    def rlgc(self, freq):
        """Frequency-dependent RLGC parameters."""
        f = np.maximum(freq, 1.0)  # Avoid division by zero

        # Skin effect: R increases as sqrt(f)
        R = self.R0 * (1 + np.sqrt(f / self.f_skin))

        # Inductance (slight decrease at high freq due to skin effect)
        L = self.L0 * np.ones_like(f)

        # Dielectric loss: G proportional to f
        G = self.G0 + 2 * np.pi * f * self.C0 * 0.002  # tan_delta = 0.002

        # Capacitance (nearly constant)
        C = self.C0 * np.ones_like(f)

        return R, L, G, C

    def propagation_params(self, freq):
        """Compute γ(f) and Z0(f)."""
        R, L, G, C = self.rlgc(freq)
        omega = 2 * np.pi * freq

        Z_series = R + 1j * omega * L
        Y_shunt = G + 1j * omega * C

        gamma = np.sqrt(Z_series * Y_shunt)
        Z0 = np.sqrt(Z_series / Y_shunt)

        # Ensure correct branch (Re(γ)>0 for attenuation)
        gamma = np.where(gamma.real < 0, -gamma, gamma)

        return gamma, Z0

    def s_parameters(self, freq, Z_ref=50.0):
        """
        Compute S-parameters using ABCD matrix.

        ABCD for transmission line of length L:
        A = cosh(γL)
        B = Z0 sinh(γL)
        C = sinh(γL)/Z0
        D = cosh(γL)
        """
        gamma, Z0 = self.propagation_params(freq)
        gL = gamma * self.length

        A = np.cosh(gL)
        B = Z0 * np.sinh(gL)
        C = np.sinh(gL) / Z0
        D = np.cosh(gL)

        # ABCD to S-parameters
        denom = A + B/Z_ref + C*Z_ref + D
        S11 = (A + B/Z_ref - C*Z_ref - D) / denom
        S21 = 2.0 / denom
        S12 = S21  # Reciprocal
        S22 = (-A + B/Z_ref - C*Z_ref + D) / denom

        return S11, S21, S12, S22

# ============================================================
# Eye Diagram Generator
# ============================================================
def generate_eye_diagram(S21, freq, data_rate=28e9, n_bits=2000, n_ui=2):
    """
    Generate eye diagram from channel S21 response.

    1. Create random NRZ bit stream
    2. Apply channel response (frequency domain)
    3. Overlay bit periods to form eye
    """
    dt = 1.0 / (2 * freq[-1])
    n_fft = 2 * (len(freq) - 1)
    t = np.arange(n_fft) * dt
    T_bit = 1.0 / data_rate

    # Random bit stream
    np.random.seed(42)
    bits = np.random.randint(0, 2, n_bits)

    # Create time-domain signal (NRZ with rise/fall time)
    samples_per_bit = int(T_bit / dt)
    if samples_per_bit < 2:
        samples_per_bit = 2
    n_samples = n_bits * samples_per_bit
    signal = np.zeros(n_samples)
    for i, bit in enumerate(bits):
        signal[i*samples_per_bit:(i+1)*samples_per_bit] = bit * 2 - 1

    # Apply raised-cosine rise/fall
    rise_samples = max(1, samples_per_bit // 5)
    rise = 0.5 * (1 - np.cos(np.pi * np.arange(rise_samples) / rise_samples))
    for i in range(1, n_bits):
        if bits[i] != bits[i-1]:
            idx = i * samples_per_bit
            if bits[i] > bits[i-1]:  # rising
                signal[idx:idx+rise_samples] = -1 + 2*rise
            else:  # falling
                signal[idx:idx+rise_samples] = 1 - 2*rise

    # Truncate/pad to FFT length
    if len(signal) > n_fft:
        signal = signal[:n_fft]
    else:
        signal = np.pad(signal, (0, n_fft - len(signal)))

    # Apply channel response
    S_fft = np.fft.rfft(signal)
    # Interpolate S21 to match FFT frequencies
    freq_fft = np.fft.rfftfreq(n_fft, d=dt)
    S21_interp = np.interp(freq_fft, freq, np.abs(S21)) * \
                 np.exp(1j * np.interp(freq_fft, freq, np.unwrap(np.angle(S21))))
    output = np.fft.irfft(S_fft * S21_interp)

    # Create eye diagram
    eye_samples = int(n_ui * T_bit / dt)
    if eye_samples < 2:
        eye_samples = 2

    n_traces = min(200, n_bits - n_ui)
    eye_traces = []
    for i in range(n_traces):
        start = i * samples_per_bit
        if start + eye_samples <= len(output):
            trace = output[start:start+eye_samples]
            eye_traces.append(trace)

    t_eye = np.linspace(0, n_ui, eye_samples) if eye_samples > 0 else np.array([0])

    return t_eye, eye_traces

# ============================================================
# TDR Simulation
# ============================================================
def simulate_tdr(freq, Z0_profile_func, Z_ref=50.0):
    """
    TDR: send step function, observe reflections.
    Z0_profile_func: returns impedance at position x.
    """
    n_freq = len(freq)
    n_fft = 2 * (n_freq - 1)
    dt = 1.0 / (2 * freq[-1])
    t = np.arange(n_fft) * dt

    # Step function (with bandwidth limit)
    step_fft = np.zeros(n_freq, dtype=complex)
    step_fft[0] = 0.5
    step_fft[1:] = 1.0 / (1j * 2 * np.pi * freq[1:])
    # Bandwidth limit
    bw = freq[-1] * 0.8
    step_fft *= np.exp(-(freq / bw)**4)

    step = np.fft.irfft(step_fft)

    # Simple TDR model: multiple impedance sections
    n_sections = 50
    x = np.linspace(0, 0.15, n_sections)
    Z_profile = np.array([Z0_profile_func(xi) for xi in x])

    # Compute reflection at each interface
    reflected = np.zeros(n_fft)
    for i in range(1, n_sections):
        Z_left = Z_profile[i-1]
        Z_right = Z_profile[i]
        rho = (Z_right - Z_left) / (Z_right + Z_left)
        delay_samples = int(2 * x[i] / (c0 * 0.6) / dt)  # v = 0.6c
        if delay_samples < n_fft:
            reflected[delay_samples:] += rho * step[:n_fft - delay_samples] * 0.3

    tdr_signal = step + reflected
    # Convert to impedance
    tdr_Z = Z_ref * (1 + reflected[:n_fft//4]) / (1 - reflected[:n_fft//4] + 1e-10)
    t_tdr = t[:n_fft//4] * 1e9  # [ns]

    return t_tdr, tdr_Z, Z_profile, x

# ============================================================
# PDN (Power Distribution Network) Impedance
# ============================================================
def compute_pdn_impedance(freq, V_dd=1.0, I_max=10.0, ripple=0.05):
    """
    PDN impedance model with VRM, bulk caps, decoupling caps, package.

    Target impedance: Z_target = V_dd * ripple / I_max
    """
    omega = 2 * np.pi * freq
    Z_target = V_dd * ripple / I_max

    # VRM (Voltage Regulator Module): R + L
    R_vrm = 0.001  # 1 mΩ
    L_vrm = 5e-9   # 5 nH
    Z_vrm = R_vrm + 1j * omega * L_vrm

    # Bulk capacitor: 100 μF with ESR and ESL
    C_bulk = 100e-6
    R_bulk = 0.01   # 10 mΩ ESR
    L_bulk = 2e-9   # 2 nH ESL
    Z_bulk = 1/(1j*omega*C_bulk) + R_bulk + 1j*omega*L_bulk

    # Package + PCB inductance
    L_pkg = 0.5e-9  # 0.5 nH
    Z_pkg = 1j * omega * L_pkg

    # Without decoupling
    Z_no_decap = 1.0 / (1.0/Z_vrm + 1.0/Z_bulk) + Z_pkg

    # Decoupling capacitors (multiple)
    # 10 × 100nF MLCC
    C_dec1 = 100e-9 * 10
    R_dec1 = 0.005
    L_dec1 = 0.3e-9
    Z_dec1 = 1/(1j*omega*C_dec1) + R_dec1 + 1j*omega*L_dec1

    # 20 × 10nF MLCC (high frequency)
    C_dec2 = 10e-9 * 20
    R_dec2 = 0.003
    L_dec2 = 0.1e-9
    Z_dec2 = 1/(1j*omega*C_dec2) + R_dec2 + 1j*omega*L_dec2

    Z_with_decap = 1.0 / (1.0/Z_vrm + 1.0/Z_bulk + 1.0/Z_dec1 + 1.0/Z_dec2) + Z_pkg

    return Z_no_decap, Z_with_decap, Z_target

# ============================================================
# Visualization
# ============================================================
def plot_results(tline, freq, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Signal Integrity / Power Integrity Analysis — High-Speed ATE (Advantest)',
                 fontsize=15, fontweight='bold')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    gamma, Z0 = tline.propagation_params(freq)
    S11, S21, S12, S22 = tline.s_parameters(freq)

    # (a) S-parameters
    ax1 = fig.add_subplot(gs[0, 0])
    f_ghz = freq / 1e9
    ax1.plot(f_ghz, 20*np.log10(np.abs(S21)+1e-15), 'b-', linewidth=2, label='S21 (IL)')
    ax1.plot(f_ghz, 20*np.log10(np.abs(S11)+1e-15), 'r-', linewidth=2, label='S11 (RL)')
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('(a) S-Parameters vs Frequency')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-60, 5)
    ax1.axhline(-3, color='gray', linestyle=':', alpha=0.5)
    ax1.text(f_ghz[-1]*0.7, -2, '-3 dB', fontsize=9, color='gray')

    # (b) Eye diagram
    ax2 = fig.add_subplot(gs[0, 1])
    t_eye, eye_traces = generate_eye_diagram(S21, freq, data_rate=28e9)
    for trace in eye_traces:
        ax2.plot(t_eye[:len(trace)], trace, 'b-', alpha=0.03, linewidth=0.5)
    ax2.set_xlabel('Time [UI]')
    ax2.set_ylabel('Amplitude [V]')
    ax2.set_title('(b) Eye Diagram (28 Gbps NRZ)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2)

    # (c) TDR impedance profile
    ax3 = fig.add_subplot(gs[0, 2])
    def impedance_profile(x):
        """Impedance vs position: 50Ω line with connector and via discontinuities."""
        if x < 0.02:
            return 50.0      # PCB trace
        elif x < 0.025:
            return 65.0      # Connector (impedance bump)
        elif x < 0.05:
            return 50.0      # PCB trace
        elif x < 0.055:
            return 35.0      # Via (impedance dip)
        elif x < 0.10:
            return 50.0      # PCB trace
        elif x < 0.105:
            return 55.0      # Package lead
        else:
            return 50.0      # DUT

    t_tdr, tdr_Z, Z_prof, x_prof = simulate_tdr(freq, impedance_profile)
    ax3.plot(t_tdr, tdr_Z, 'b-', linewidth=2)
    ax3.axhline(50, color='green', linestyle='--', alpha=0.5, label='50 Ω target')
    ax3.set_xlabel('Time [ns]')
    ax3.set_ylabel('Impedance [Ω]')
    ax3.set_title('(c) TDR Impedance Profile')
    ax3.set_ylim(20, 80)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Annotate discontinuities
    ax3.annotate('Connector', xy=(0.15, 58), fontsize=9, color='red')
    ax3.annotate('Via', xy=(0.35, 38), fontsize=9, color='red')
    ax3.annotate('Package', xy=(0.6, 56), fontsize=9, color='red')

    # (d) Characteristic impedance vs frequency
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(f_ghz, np.real(Z0), 'b-', linewidth=2, label='Re(Z0)')
    ax4.plot(f_ghz, np.imag(Z0), 'r--', linewidth=2, label='Im(Z0)')
    ax4.set_xlabel('Frequency [GHz]')
    ax4.set_ylabel('Impedance [Ω]')
    ax4.set_title('(d) Characteristic Impedance Z₀(f)')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(50, color='green', linestyle=':', alpha=0.5)

    # (e) Loss decomposition
    ax5 = fig.add_subplot(gs[1, 1])
    alpha = gamma.real  # Attenuation [Np/m]
    alpha_dB = 20 * alpha * tline.length / np.log(10)  # Total loss in dB

    R, L, G, C = tline.rlgc(freq)
    omega = 2 * np.pi * freq
    # Conductor loss
    alpha_c = R / (2 * np.sqrt(L/C).real)
    alpha_c_dB = 20 * alpha_c * tline.length / np.log(10)
    # Dielectric loss
    alpha_d = G * np.sqrt(L/C).real / 2
    alpha_d_dB = 20 * alpha_d * tline.length / np.log(10)

    ax5.plot(f_ghz, alpha_dB, 'k-', linewidth=2, label='Total loss')
    ax5.plot(f_ghz, alpha_c_dB, 'b--', linewidth=1.5, label='Conductor loss')
    ax5.plot(f_ghz, alpha_d_dB, 'r--', linewidth=1.5, label='Dielectric loss')
    ax5.set_xlabel('Frequency [GHz]')
    ax5.set_ylabel('Insertion Loss [dB]')
    ax5.set_title('(e) Loss Decomposition')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # (f) PDN impedance
    ax6 = fig.add_subplot(gs[1, 2])
    freq_pdn = np.logspace(3, 9, 1000)
    Z_no, Z_with, Z_target = compute_pdn_impedance(freq_pdn)
    ax6.loglog(freq_pdn/1e6, np.abs(Z_no), 'r-', linewidth=2,
              label='Without decoupling caps')
    ax6.loglog(freq_pdn/1e6, np.abs(Z_with), 'b-', linewidth=2,
              label='With decoupling caps')
    ax6.axhline(Z_target, color='green', linestyle='--', linewidth=2,
               label=f'Target Z = {Z_target*1000:.0f} mΩ')
    ax6.set_xlabel('Frequency [MHz]')
    ax6.set_ylabel('|Z_PDN| [Ω]')
    ax6.set_title('(f) PDN Impedance (Power Integrity)')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, which='both')
    ax6.set_xlim(1e-3, 1e3)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("Signal Integrity / Power Integrity Analysis")
    print("Portfolio: Advantest Corporation")
    print("=" * 60)

    # Transmission line parameters
    tline = TransmissionLine(
        R0=2.0,        # 2 Ω/m DC resistance
        L0=250e-9,     # 250 nH/m inductance
        G0=0.0,        # Negligible DC conductance
        C0=100e-12,    # 100 pF/m capacitance
        f_skin=1e9,    # Skin effect onset at 1 GHz
        length=0.10    # 10 cm trace length
    )

    # Z0 at DC
    Z0_dc = np.sqrt(tline.L0 / tline.C0)
    print(f"\nTransmission line: length={tline.length*100:.0f} cm")
    print(f"  Z0 (lossless) = {Z0_dc:.1f} Ω")
    print(f"  Velocity = {1/np.sqrt(tline.L0*tline.C0)/c0*100:.1f}% of c")

    # Frequency range
    freq = np.linspace(0.1e9, 50e9, 2000)

    # Compute
    gamma, Z0 = tline.propagation_params(freq)
    S11, S21, S12, S22 = tline.s_parameters(freq)

    # -3dB bandwidth
    IL_dB = 20 * np.log10(np.abs(S21) + 1e-15)
    bw_idx = np.argmin(np.abs(IL_dB - (-3)))
    print(f"  -3 dB bandwidth: {freq[bw_idx]/1e9:.1f} GHz")
    print(f"  Loss at 14 GHz (Nyquist for 28G): {np.interp(14e9, freq, IL_dB):.1f} dB")

    # PDN target impedance
    Z_target = 1.0 * 0.05 / 10.0
    print(f"\nPDN target impedance: {Z_target*1000:.0f} mΩ (Vdd=1V, Imax=10A, 5% ripple)")

    # Plot
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "signal_integrity_results.png")
    print("\nPlotting results...")
    plot_results(tline, freq, save_path)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")
