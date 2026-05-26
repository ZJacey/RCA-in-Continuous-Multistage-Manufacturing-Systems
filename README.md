# RCA-in-Continuous-Multistage-Manufacturing-Systems

This repository provides the code and data implementation for the AD-STGN model, designed for Root Cause Analysis (RCA) in complex, nonlinear, and highly coupled continuous manufacturing processes.
## Datasets Overview
To validate the effectiveness of the framework, experiments are conducted on two widely-recognized cyber-physical benchmarks and one real-world industrial case study.
### 1. Tennessee Eastman Process (TEP)
The TEP dataset is a widely adopted benchmark for RCA, simulating a complex, nonlinear continuous process.
* **System Network:** Spatial graph comprising $N=51$ topological nodes (40 continuous sensor measurements, 11 manipulated control actions).
* **Training Set:** 230,500 steady-state samples (used exclusively to prevent data leakage and for causal graph extraction).
* **Testing Set:** Mixed sequences constructed by stitching 1,081 steady-state testing samples and 800 faulty testing samples.
### 2. Secure Water Treatment (SWaT)
Operational data collected from a raw water purification plant across six physical stages (raw water supply, chemical dosing, ultrafiltration, dechlorination, reverse osmosis, and backwash).
* **System Network:** $N=50$ input nodes (24 continuous sensors, 26 discrete actuators) and 1 prediction target.
* **Training Set:** 496,800 steady-state samples representing normal behavior (collected over 7 days).
* **Testing Set:** 449,919 samples containing anomalous events (collected over 4 days of cyber-physical attacks).
### 3. Real-world Case Study: Injection Molding Production Line
Deployed on a real-world injection molding production line of a leading global Automotive Electronics manufacturer in Tianjin. Data was collected from Manufacturing Execution Systems and IoT sensors between February 2, 2025, and March 14, 2025.
* **Process Stages:** Clamping, injection, holding, cooling, ejection, and robot picking/placing.
* **System Network:** 66 continuous sensor measurements (granular process dynamics) and 7 discrete control actions (machine setpoints and equipment configurations).
* **Terminal State Indicator:** $X_{66}$ (part temperature when placed by the robotic arm). It integrates the cumulative thermal-mechanical history. Deviations correlate with terminal Injection Molding Process Defect (PMT) issues (e.g., internal bubbles, warpage, incomplete cooling).
* **Data Split:** * **Training Set:** 88,000 normal steady-state samples.
  * **Validation Set:** 22,614 normal samples.
  * **Testing Set:** Chronological sequence where anomalies start from the 75th sample, representing a transition from normal operation to a PMT outbreak.
