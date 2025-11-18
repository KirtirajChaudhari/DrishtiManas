# Drishti Manas Roadmap

## Why a roadmap?
Ophthalmologists told us they need trustworthy automation aligned with current clinic workflows. This document translates inspirations from the reference repositories into concrete delivery milestones.

## Reference inspirations

### 1. Dry-Eye-Disease (owais825)
- **Focus on tear film quality and blink artifacts.** We will port their pre-processing heuristics into our technician QC checklist.
- **Symptom capture matters.** Keep the patient questionnaire (dry eye severity, screen time) adjacent to image uploads.

### 2. Prediction of Ocular Disease via Fundus CNN (Nithish-2002)
- **Multi-label CNN ensembles.** Their ROC-driven label smoothing informs our plan to maintain shared fundus encoders with disease-specific heads.
- **Explainability overlay.** Grad-CAM examples from this repo shape how we annotate hotspots for each prediction card.

### 3. Ocular Disease Identifier (DSC-McMaster-U)
- **Clinician worklist UX.** Their tabular review queue influenced our doctor dashboard layout and triage chips.
- **Model governance.** The repo emphasizes logging/metrics; we extend that into audit logging and version-control hooks.

## Delivery milestones

| Quarter | Milestone | Notes |
| --- | --- | --- |
| Q1 | Dry-eye friendly upload wizard | Integrate blink-detection, enforce illumination hints, connect symptom questionnaire. |
| Q1 | CNN ensemble baseline | Port multi-task fundus model with disease-specific thresholds inspired by repos 2 & 3. |
| Q2 | Explainable reporting pack | Auto-generate PDF with Grad-CAM, triage, and documentation snippets per doctor feedback. |
| Q2 | Audit + admin console | Expand admin role to review model versions, drawing from DSC-McMaster-U governance ideas. |
| Q3 | Neurology module GA | Extend pipelines to neurological screenings with shared UI patterns. |

## Immediate next steps
1. Harden FastAPI endpoints with auth + persistence.
2. Replace placeholder healthmaps with true Grad-CAM overlays from the PyTorch core.
3. Launch dockerized demo stack (backend, frontend, Postgres) for clinic pilots.
