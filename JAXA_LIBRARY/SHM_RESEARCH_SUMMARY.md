# JAXA Structural Health Monitoring (SHM) Research Context

**Summary of findings on JAXA's approach to SHM and Composite Structures.**

## 1. Research Focus Areas
JAXA's research into structural safety and SHM focuses on:
1.  **Composite Impact Damage**: Detecting Barely Visible Impact Damage (BVID) in CFRP structures (fuselages, tanks, fairings).
2.  **Bondline Integrity**: Monitoring the quality of adhesive bonds in composite joints and sandwich structures (skin-core bonding).
3.  **Life Cycle Monitoring**: From manufacturing (process monitoring) to operation (launch loads).

## 2. Key Technologies
### 2.1 Ultrasonic Guided Waves (Lamb Waves)
*   **Principle**: Exciting plate waves ($S_0$, $A_0$ modes) that propagate long distances in thin-walled structures.
*   **Application**: Large area monitoring of fairings and tanks.
*   **Challenge**:
    *   **Mode Selection**: $S_0$ is faster and less dispersive but harder to excite in sandwich panels. $A_0$ is sensitive to core damage but highly attenuated.
    *   **Anisotropy**: Wave velocity depends on propagation angle relative to fiber direction.
*   **JAXA Relevance**: Research into "Smart Skins" with embedded fiber optic sensors (FBG) or piezoelectric (PZT) networks.

### 2.2 Acoustic Emission (AE)
*   **Usage**: Passive monitoring of crack initiation (matrix cracking, fiber breakage) during proof testing or flight.
*   **Limitation**: Requires active damage growth to detect; susceptible to noise during launch (acoustic environment).

### 2.3 Fiber Bragg Grating (FBG) Sensors
*   **Advantage**: EMI immune, lightweight, multiplexable.
*   **Application**: Strain monitoring and impact detection on cryogenic tanks (H3 LH2 tanks).

## 3. Structural Design Standards (JAXA-JMR-002/JERG)
*   **Damage Tolerance**: JAXA requires safety-critical composite structures to demonstrate "No Growth" of defects under service loads if they cannot be inspected.
*   **SHM Role**: SHM can potentially replace periodic NDI (Non-Destructive Inspection) or allow for "Condition-Based Maintenance" (CBM), reducing ground operations costs (a key goal for Epsilon/H3).

## 4. Specific Context for Payload Fairings
*   **Material**: CFRP Face Sheets + Aluminum Honeycomb Core.
*   **Defect Types**:
    *   **Disbond/Delamination**: Caused by manufacturing flaws or acoustic fatigue.
    *   **Water Ingress**: Moisture trapped in honeycomb cells (detected by localized damping changes).
*   **Acoustic Fatigue**: The fairing endures ~147 dB acoustic loads. SHM systems must distinguish between "noise" (structural vibration) and "signal" (defect scattering) or operate during quiet phases (pre-launch/orbit).

## 5. References (General Literature)
*   *Rose, J. L.*: "Ultrasonic Guided Waves in Solid Media" (Standard text, widely cited by JAXA researchers).
*   *Su, Z. & Ye, L.*: "Identification of Damage Using Lamb Waves".
*   *JAXA Research Publications*: JAXA often publishes in *Composite Structures* and *Structural Health Monitoring* journals regarding "impact damage suppression" and "reliable bonding".
