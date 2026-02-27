# JAXA Technical Document Library

This directory contains technical documents, standards, and research summaries relevant to the **Payload Fairing SHM Project** for JAXA Epsilon and H3 rockets.

## 1. Official JAXA Documents (Downloaded PDFs)

| Filename | Title | Description |
| :--- | :--- | :--- |
| **[EpsilonUsersManual_e.pdf](./EpsilonUsersManual_e.pdf)** | **Epsilon Launch Vehicle User's Manual** | **Primary Reference**. Detailed specs on Epsilon payload environment (acoustic, shock, thermal), fairing dimensions, and interfaces. |
| **[JAXA-JMR-002E.pdf](./JAXA-JMR-002E.pdf)** | **Launch Vehicle Payload Safety Standard** | Safety requirements for payloads. Defines structural safety factors (SF > 1.4/1.5) and hazard control for pressurized/explosive systems. |
| **[JAXA-JMR-001C.pdf](./JAXA-JMR-001C.pdf)** | **System Safety Standard** | High-level safety program requirements for JAXA projects. |
| **[JAXA-JERG-1-007E.pdf](./JAXA-JERG-1-007E.pdf)** | **Safety Regulation for Launch Site Operation** | Operational safety rules at Uchinoura (Epsilon) and Tanegashima (H3). |
| **[KHI_Fairing_Tech_Desc.pdf](./KHI_Fairing_Tech_Desc.pdf)** | **Kawasaki Heavy Industries Fairing Tech** | Technical description of the fairing separation mechanism ("clamshell" type) and manufacturing by KHI. |

## 2. Research Summaries (Generated)

| Filename | Content |
| :--- | :--- |
| **[H3_ROCKET_SPECS.md](./H3_ROCKET_SPECS.md)** | **H3 Rocket Specifications**. Compiled details on H3 fairing types (S/L/W), SRB-3 composite structure, and LE-9 engine, as no official PDF manual is public. |
| **[SHM_RESEARCH_SUMMARY.md](./SHM_RESEARCH_SUMMARY.md)** | **SHM Research Context**. Summary of Guided Wave and Acoustic Emission technologies applicable to JAXA composite structures. |

## 3. Key Technical Insights for SHM

*   **Acoustic Environment**: The fairing experiences **~147 dB** acoustic pressure at lift-off. This is the primary driver for "Acoustic Fatigue" which can cause skin-core debonding.
*   **Separation Shock**: The "Low-Shock Separation System" (Notched Bolt) reduces shock levels significantly compared to older explosive bolts, but monitoring separation events remains critical.
*   **Material**: Both Epsilon and H3 fairings use **CFRP Face Sheets + Aluminum Honeycomb Core**. This sandwich structure is challenging for ultrasound (high attenuation) but ideal for stiffness/mass ratio.
