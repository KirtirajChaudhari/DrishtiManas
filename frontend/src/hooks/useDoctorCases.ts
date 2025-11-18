import { useMemo } from "react";

import type { PredictionResponse } from "../types/prediction";

const demoCases: PredictionResponse[] = Array.from({ length: 6 }).map((_, index) => ({
  id: `demo-${index + 1}`,
  patient_id: `MRN-${1000 + index}`,
  created_at: new Date(Date.now() - index * 60 * 60 * 1000).toISOString(),
  results: [
    {
      disease: "Glaucoma",
      probability: 0.4 + Math.random() * 0.4,
      triage: index % 2 === 0 ? "elevated" : "watch",
      heatmap: { url: null, caption: null }
    }
  ]
}));

export function useDoctorCases() {
  return useMemo(() => ({
    cases: demoCases,
    metrics: {
      totalCases: demoCases.length,
      criticalCases: demoCases.filter((c) => c.results[0].triage === "critical").length,
      sla: "18 min"
    }
  }), []);
}
