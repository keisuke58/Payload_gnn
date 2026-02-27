# JAXA Rocket Structural Engineering & SHM Wiki

This knowledge base consolidates deep technical details on JAXA's Epsilon and H3 launch vehicles, focusing on structural dynamics, environmental loads, and the necessity for advanced Structural Health Monitoring (SHM).

## 1. Launch Vehicle Structural Systems

### 1.1 Payload Fairing (PLF)
The payload fairing protects the satellite from aerodynamic forces, acoustic noise, and thermal loads during ascent.

*   **Materials**:
    *   **Face Sheets**: Carbon Fiber Reinforced Polymer (CFRP). Typically Toray T1000 or equivalent high-modulus fibers.
    *   **Core**: Aluminum Honeycomb (Al 5052/5056). Provides high bending stiffness with minimal mass.
    *   **Construction**: Automated Tape Laying (ATL) or Filament Winding on female molds. Manufactured by **Kawasaki Heavy Industries (KHI)** (S/L types) and **Beyond Gravity** (W type for H3).
*   **Separation Mechanism**:
    *   **Type**: "Clamshell" opening (split into two halves).
    *   **Actuation**: **Low-Shock Separation System** developed by KHI.
    *   **Mechanism**: Uses **Notched Bolts** (Frangible Bolts) expanded by mild detonating fuse (MDF) or non-explosive actuators, rather than high-shock explosive bolts. This reduces shock loads ($< 1000 \text{ G}$ at interface) to sensitive payloads.
    *   **Hinge**: Passive rotation hinges with spring actuators.
*   **Venting System**:
    *   **Purpose**: Equalize internal pressure with rapidly dropping ambient pressure during ascent to prevent structural ballooning or collapse.
    *   **Critical Phase**: Transonic flight and Max Q, where $\Delta P$ is highest.

### 1.2 Interstage & Motor Case Structures
*   **Solid Motor Cases (Epsilon/SRB-3)**:
    *   **Material**: CFRP Filament Winding (FW). Monolithic composite structure.
    *   **Load Path**: Carries axial thrust and bending moments.
*   **Interstage Adapters**:
    *   **Design**: Often **CFRP Lattice Shells** (Grid stiffened) or **Aluminum Honeycomb Sandwich**.
    *   **Function**: Connects stages (e.g., 2nd to 3rd) and houses avionics. High buckling strength required.

---

## 2. Launch Environmental Loads (The SHM Context)

Understanding these loads is crucial for defining SHM requirements (i.e., *when* does damage occur?).

### 2.1 Acoustic Loads (Lift-off & Transonic)
*   **Source**: Rocket exhaust jet mixing with air (Lift-off) and shock wave interaction (Transonic).
*   **Magnitude**:
    *   **External (Surface)**: $\sim 147 \text{ dB}$ (OASPL).
    *   **Internal (Payload)**: $\sim 140 \text{ dB}$ (attenuated by acoustic blankets).
*   **SHM Relevance**: High-cycle fatigue. Can cause **skin-core debonding** in sandwich panels due to local resonance of the face sheet (breathing mode).

### 2.2 Aerodynamic Loads (Max Q)
*   **Max Q (Maximum Dynamic Pressure)**: Occurs $\sim 60\text{s}$ after lift-off.
    *   Formula: $q = \frac{1}{2} \rho v^2$
    *   **Epsilon/H3**: Typically $30 \sim 40 \text{ kPa}$.
*   **Buffeting**: Unsteady oscillating pressure at transonic speeds (Mach 0.8 - 1.2).
*   **SHM Relevance**: Static compression combined with oscillating bending loads. Critical for **buckling** and propagation of existing delaminations.

### 2.3 Separation Shock (Pyro Shock)
*   **Events**: SRB separation, Fairing jettison, Stage separation.
*   **Characteristics**: High frequency ($100 \text{ Hz} - 10 \text{ kHz}$), high amplitude ($> 2000 \text{ G}$ near source), short duration (< 10 ms).
*   **H3 Incident Context**: The H3 TF1 failure involved a suspected electrical anomaly possibly triggered by separation shock or interstage interference, highlighting the criticality of monitoring separation events.

### 2.4 Thermal Loads
*   **Source**: Aerodynamic friction (Free molecular flow heating at high altitude).
*   **Tip Temperature**: Can exceed $500^\circ\text{C}$ at the nose cone tip (protected by ablator/insulator).
*   **Fairing Skin**: $\sim 100-200^\circ\text{C}$.
*   **SHM Relevance**: Thermal expansion mismatch between CFRP skin and Al core can induce shear stresses, promoting debonding.

---

## 3. Structural Health Monitoring (SHM) Strategy

### 3.1 Target Defect: Skin-Core Debonding
*   **Mechanism**: Separation of the CFRP face sheet from the Aluminum honeycomb core.
*   **Cause**: Impact (ground handling), Acoustic Fatigue (launch), Manufacturing defects (voids).
*   **Consequence**: Loss of local buckling stability $\rightarrow$ Catastrophic fairing failure (breakup).

### 3.2 Guided Wave-Based Detection (Our Approach)
*   **Physics**:
    *   **Lamb Waves**: $A_0$ (Flexural) and $S_0$ (Extensional) modes.
    *   **Feature**: In a sandwich panel, high-frequency waves ($>50 \text{ kHz}$) confined to the skin are sensitive to debonding.
    *   **Anomaly**: Wave trapping (standing waves) and amplitude amplification in the detached skin region.

### 3.3 GNN Implementation for Epsilon/H3
*   **Geometry**: 1/6th Cylindrical Shell (Symmetry utilized).
*   **Graph Nodes**: PZT Sensor network coordinates $(x, y, z)$.
*   **Edge Features**: Geodesic distance on the curved fairing surface.
*   **Input**: Sensor time-series data (Strain/Voltage).
*   **Output**: Probability of debonding at each graph node.

---

## 4. Reference Specifications (Estimated)

| Parameter | Epsilon S (Enhanced) | H3 (Standard) |
| :--- | :--- | :--- |
| **Fairing Diameter** | $\sim 2.6 \text{ m}$ | $5.2 \text{ m}$ (Type-W) |
| **Fairing Length** | $\sim 9 \text{ m}$ | $\sim 16 \text{ m}$ (Long) |
| **Material** | CFRP/Al-Honeycomb | CFRP/Al-Honeycomb |
| **Max Acoustic Load** | $147 \text{ dB}$ (Ext) | $148 \text{ dB}$ (Ext) |
| **Separation System** | Notched Bolt (Low Shock) | Notched Bolt (Low Shock) |
| **Orbit** | SSO / LEO | GTO / SSO |

---

## 5. Advanced Engineering Deep Dive (Beyond Standard Knowledge)

This section contains expert-level details on JAXA's specific engineering approaches, exceeding typical public knowledge.

### 5.1 Acoustic Suppression "Flue" Technology
*   **Context**: The acoustic environment at lift-off is the most severe mechanical load for the payload.
*   **Innovation**: JAXA developed a specific **Acoustic Suppression Flue** (sound tunnel) for the Epsilon launch pad.
*   **Performance**: This geometric innovation reduced acoustic vibrations acting on the payload to **less than 1/10th** of the levels experienced by the previous M-V rocket.
*   **Significance**: This allows Epsilon to carry more sensitive scientific instruments without heavy damping structures.

### 5.2 Structural Design Criteria & Safety Factors
JAXA aligns with international aerospace standards for structural safety, specifically **JMR-002 (Launch Vehicle Payload Safety Standard)**.
*   **Metallic Structures**:
    *   **Yield Safety Factor**: $1.1 \sim 1.25$ (Limit Load $\times$ Factor $\le$ Yield Strength)
    *   **Ultimate Safety Factor**: $1.4 \sim 1.5$ (Limit Load $\times$ Factor $\le$ Ultimate Strength)
*   **Composite/Bonded Structures (CFRP)**:
    *   **Ultimate Safety Factor**: Typically **1.5** (Unmanned) to **2.0** (Discontinuities/Joints) as per JMR-002.
    *   **Special Condition**: For bonded joints (like the fairing skin-to-core), a "No-Growth" criterion for defects is applied under service loads, requiring SHM to verify that defects remain below critical size.

### 5.3 Interstage & Adapter Construction
*   **Standard Design**: Aluminum Honeycomb Sandwich with CFRP Face Sheets (similar to fairings) is the baseline for Epsilon/H3 interstages to minimize mass ($\sim 730 \text{ kg}$ for large adapters).
*   **Advanced Research (Isogrid/Anisogrid)**:
    *   JAXA and European partners (Vega-C) are actively researching **CFRP Anisogrid Lattice** structures.
    *   **Benefit**: These "rib-stiffened" open lattice structures offer superior buckling resistance per unit mass compared to monolithic shells and eliminate the risk of sandwich core debonding/water ingress.
    *   **SHM Implication**: Monitoring a lattice structure requires tracking rib buckling modes rather than skin delamination.

### 5.4 Epsilon S & H3 Synergy
*   **SRB-3 Commonality**: The Epsilon S First Stage is a modified **SRB-3** (Solid Rocket Booster) from the H3 rocket. This drastically reduces manufacturing costs through economies of scale.
*   **Mobile Launch Control**:
    *   **Revolutionary Operation**: Epsilon launch operations are streamlined to require only **8 personnel** (vs. 150 for M-V).
    *   **Implication**: The "intelligence" is moved to the rocket itself (autonomous checkout), reducing ground infrastructure needs. This is a key "Epsilon" philosophy: "A Rocket with Artificial Intelligence."

### 5.5 Manufacturing Innovations
*   **Simplified Molding**: Epsilon's motor cases (2nd/3rd stage) use CFRP with a simplified, lower-cost molding process compared to the M-V's expensive high-performance composites.
*   **Material**: High-strength Carbon Fiber (Toray T1000 equivalent) allows the 2nd stage to be lighter and perform better, contributing to the payload capacity increase.

