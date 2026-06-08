# Stirling Engine Simulator V10 — Stable Notes

## Engineering Story

Stage 1 — Baseline + Regenerator Study  
Understand the original prototype and quantify the effect of the regenerator.

Stage 2 — Constrained Geometry Optimization / Prototype 2  
Improve the engine while keeping the existing displacer fixed.

Stage 3 — Operating Conditions on Prototype 2  
Optimize gas, mean pressure, frequency, and hot-side temperature using the Prototype 2 geometry.

Stage 4 — Full Free Geometry Optimization  
Explore the theoretical design-space limit when the main geometry is allowed to change freely.

## Key Features Added

- Correct no-regenerator model using minimal connecting pipe volume.
- Prototype 1 vs Prototype 2 performance comparison.
- Regenerator impact tables showing required heat input with and without regenerator.
- Stage 3 operating-condition graphs:
  - Power vs efficiency operating map
  - Brake power by gas and objective
  - Selected pressure and frequency
  - Heat budget usage
  - Power vs pressure by gas and prototype
- Stage 4 full-geometry comparison:
  - Prototype 1
  - Prototype 2
  - Max Power
  - Max Efficiency
  - Balanced
- Stage 4 regenerator impact.
- Engineering story box in the optimization tab.
- Deprecated Streamlit width arguments cleaned.

## Presentation Logic

Prototype 2 is the practical redesign.  
Stage 4 is not the manufacturing proposal; it is a theoretical benchmark for future redesign.

