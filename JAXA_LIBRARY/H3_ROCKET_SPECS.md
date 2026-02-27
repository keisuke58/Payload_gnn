# JAXA H3 Launch Vehicle Specifications & Structural Details

**Note**: As of Feb 2026, a public "H3 User's Manual" PDF is not directly available. This document consolidates technical specifications from JAXA official pages, Mitsubishi Heavy Industries (MHI) data, and related technical reports.

## 1. Overview
The H3 Launch Vehicle is Japan's new flagship rocket, designed for high flexibility, high reliability, and high cost-performance. It succeeds the H-IIA/B.

*   **Developer**: JAXA & Mitsubishi Heavy Industries (MHI)
*   **Maiden Flight**: March 2023 (TF1 - Failed), Feb 2024 (TF2 - Success)
*   **Key Philosophy**: "Easy-to-use" rocket with modular configurations.

## 2. Structural Configurations
The H3 uses a naming convention "H3-abc":
*   **a (Engines)**: Number of LE-9 First Stage Engines (2 or 3).
*   **b (Boosters)**: Number of SRB-3 Solid Rocket Boosters (0, 2, or 4).
*   **c (Fairing)**: Fairing Type (S: Short, L: Long, W: Wide).

### 2.1 First Stage
*   **Engine**: LE-9 (Expander Bleed Cycle).
    *   Thrust: ~1471 kN (vacuum).
    *   Propellant: Liquid Hydrogen (LH2) / Liquid Oxygen (LOX).
*   **Structure**: Aluminum alloy tankage (likely 2219 or similar Al-Li alloys based on H-IIA heritage).

### 2.2 Solid Rocket Booster (SRB-3)
*   **Commonality**: Used as the First Stage for **Epsilon S**.
*   **Case Material**: CFRP (Carbon Fiber Reinforced Polymer) via Filament Winding.
*   **Propellant**: Composite solid propellant.
*   **Thrust**: ~2158 kN.
*   **Mounting**: Attached to the first stage core. Separation uses separation motors/struts.

### 2.3 Payload Fairing (PLF)
*   **Manufacturer**: Kawasaki Heavy Industries (KHI) (Standard/Long), Beyond Gravity (Wide).
*   **Structure**:
    *   **Face Sheet**: CFRP (Carbon Fiber Reinforced Polymer).
    *   **Core**: Aluminum Honeycomb.
    *   **Construction**: Sandwich structure cured in large autoclaves.
*   **Types**:
    *   **Type-S (Short)**: Standard missions.
    *   **Type-L (Long)**: Tall payloads.
    *   **Type-W (Wide)**: Dual launch or large diameter payloads (5.2m diameter).
*   **Separation System**:
    *   **Low-Shock System**: Uses "Notched Bolts" (Frangible Bolts) expanded by non-explosive actuators or mild detonating fuses to minimize shock to the payload.
    *   **Acoustic Protection**: Equipped with acoustic blankets to attenuate internal noise levels to < 140 dB (approx).

## 3. Launch Environment (Estimated from H-IIA & Epsilon)
Since specific H3 manuals are restricted, these values are estimated based on heritage:

### 3.1 Acoustic Environment
*   **External Level**: ~148 dB (OASPL) at lift-off.
*   **Internal Level**: Attenuated to ~135-140 dB by fairing structure and acoustic blankets.
*   **Frequency Content**: Broadband random noise, peak energy typically in 100-1000 Hz range.

### 3.2 Mechanical Environment
*   **Quasi-Static Acceleration**: Max ~3-5 G (Axial) during first stage flight.
*   **Sine Vibration**: Low frequency transients (5-100 Hz).
*   **Shock**: Separation events (SRB, Fairing, Spacecraft) generate high-frequency pyro-shock (up to several thousand Gs at source, attenuating with distance).

## 4. Relevance to SHM Research
*   **Target**: The **Payload Fairing (PLF)** is the primary target for Guided Wave SHM.
*   **Critical Failure Mode**: Debonding between CFRP skin and Al-Honeycomb core due to acoustic fatigue or impact.
*   **Geometry**: Large surface area (Type-W is >5m diameter), requiring scalable sensor networks.
*   **Material**: The CFRP/Honeycomb sandwich is highly attenuative to high-frequency ultrasound, necessitating optimized sensor placement (30-50 cm spacing estimated).

## 5. Launch History & Current Status (As of Feb 2026)

### 5.1 Launch Record
The H3 has entered its operational phase, though it faces ongoing challenges with upper stage reliability.

| Flight | Date | Outcome | Payload | Note |
| :--- | :--- | :--- | :--- | :--- |
| **TF1** | 2023/03/07 | **Failure** | ALOS-3 (Daichi-3) | 2nd stage ignition failure. Destruct command issued. |
| **TF2** | 2024/02/17 | **Success** | VEP-4, CE-SAT-IE | 2nd stage ignition confirmed. Orbit achieved. |
| **F3** | 2024/07/01 | **Success** | ALOS-4 (Daichi-4) | First operational flight success. |
| **F4** | 2024/11/04 | **Success** | DSN-3 (Kirameki-3) | Defense communication satellite. |
| **F5** | 2025/02/02 | **Success** | QZS-6 (Michibiki-6) | Quasi-Zenith Satellite. |
| **F7** | 2025/10/26 | **Success** | HTV-X1 | New cargo transporter to ISS. First use of **Type-W** fairing? |
| **F8** | 2025/12/22 | **Failure** | QZS-5 (Michibiki-5) | 2nd stage anomaly (ignition/combustion). Orbit insertion failed. |
| **F9** | (Postponed) | Pending | QZS-7 (Michibiki-7) | Delayed due to F8 failure investigation. |

*(Note: Flight F6 (H3-30S configuration) appears to have been rescheduled or delayed out of sequence)*

### 5.2 Current Issues (Feb 2026)
*   **Reliability Focus**: The failure of Flight F8 (Dec 2025) has brought renewed scrutiny to the **LE-5B-3** second stage engine, similar to the initial failure of TF1.
*   **Schedule Impact**: The launch of F9 (originally planned for Feb 2026) is currently on hold pending the investigation results.

