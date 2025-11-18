import type { PredictionResponse } from "../types/prediction";

export const mockPrediction: PredictionResponse = {
  id: "patient-001",
  patient_id: "patient-001",
  created_at: new Date().toISOString(),
  results: [
    {
      disease: "Glaucoma",
      probability: 0.83,
      triage: "elevated",
      heatmap: { url: "https://via.placeholder.com/120", caption: "Disc rim cupping" }
    },
    {
      disease: "Diabetic Retinopathy",
      probability: 0.32,
      triage: "watch",
      heatmap: { url: "https://via.placeholder.com/120", caption: "Microaneurysms" }
    }
  ]
};
