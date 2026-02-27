# Project Target Selection: Epsilon S vs H3

This document analyzes which launch vehicle is the optimal target for our **Graph Neural Network (GNN) based Structural Health Monitoring (SHM)** project.

## 1. Comparative Analysis Overview

| Feature | **Epsilon S (Enhanced)** | **H3 (Flagship)** |
| :--- | :--- | :--- |
| **Primary Concept** | "The Smart Rocket" (Autonomous Checkout) | High Flexibility, Reliability, Cost Performance |
| **Propulsion** | Solid Fuel (High Vibration) | Liquid Fuel (LE-9) + SRB-3 |
| **Fairing Diameter** | $\sim 2.6 \text{ m}$ | $5.2 \text{ m}$ (Type-S/L/W) |
| **Fairing Height** | $\sim 9 \text{ m}$ | $10.4 \text{ m}$ (S) / $16.4 \text{ m}$ (L) |
| **Acoustic Load** | High (Solid Motor Noise) | High (SRB-3 + LE-9 Interaction) |
| **SHM Synergy** | **High**: Fits "Artificial Intelligence" concept. | **Medium**: Focus is on cost reduction. |
| **Current Status** | Development (Recent Motor Test Failure) | Operational (Test Flight 2 Success) |

---

## 2. Detailed Consideration (考察)

### 2.1 Epsilon S: The "Smart" Choice (Recommended for System Logic)
**Why choose Epsilon?**
*   **"Smart Rocket" Philosophy**: Epsilon is officially designated as a rocket with "Artificial Intelligence" (autonomous checkout systems like ROSE). Integrating an onboard SHM system aligns perfectly with JAXA's roadmap for this vehicle.
*   **Environment**: Solid fuel rockets generate sharper, higher-frequency vibration environments, making acoustic fatigue a more critical concern for the fairing and motor cases.
*   **Scale**: The smaller scale ($2.6 \text{ m}$) is computationally lighter for simulation but still complex enough to demonstrate GNN capabilities.

**Risks**:
*   The recent 2nd stage motor failure (July 2023, Nov 2024) might delay the program, though it highlights the desperate need for better monitoring systems.

### 2.2 H3: The "Impact" Choice (Recommended for Scale)
**Why choose H3?**
*   **National Flagship**: H3 is Japan's primary access to space. Solving a problem for H3 has the highest industrial impact.
*   **Large-Scale Wave Propagation**: The $5.2 \text{ m}$ diameter fairing presents a massive surface area. Detecting damage across such a large honeycomb sandwich structure is a challenging problem where GNNs (which scale well with graph size) can shine over traditional wired sensors.
*   **Cost Efficiency**: H3 aims to halve launch costs. Automated inspection (SHM) replacing manual ground inspection fits this cost-reduction goal.

---

## 3. Technical Recommendation

### **Option A: Go with H3 (Current Path)**
*   **Reason**: You have already configured the simulation for $R=2600 \text{ mm}$ ($5.2 \text{ m}$ dia). The large surface area highlights the benefit of the "Sparse Sensor GNN" approach.
*   **Narrative**: "Applying advanced AI monitoring to Japan's largest next-gen launcher to ensure safety and reduce inspection costs."

### **Option B: Pivot to Epsilon**
*   **Reason**: If you want to emphasize "Autonomous Launch Operations".
*   **Narrative**: "Enhancing the 'Smart Rocket' vision with real-time structural diagnosis."

**Final Verdict**:
Unless you specifically want to focus on "Solid Rocket Motor" acoustics, **H3 is the stronger candidate** for a fairing SHM study because the sheer size of the fairing makes manual inspection difficult and wave propagation simulation more impressive.

---

## 4. Design References
*   **H3**: Uses Split-Fairing (Clamshell), CFRP Face / Al-Honeycomb Core.
*   **Epsilon**: Uses similar construction but smaller. The Epsilon S uses the H3's SRB-3 as its first stage, so they share DNA.
