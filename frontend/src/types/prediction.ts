export interface Heatmap {
  url: string | null;
  caption?: string | null;
}

export interface PredictionResult {
  disease: string;
  probability: number;
  triage: "critical" | "elevated" | "watch" | "baseline";
  heatmap: Heatmap;
}

export interface PredictionResponse {
  id: string;
  patient_id: string;
  created_at: string;
  results: PredictionResult[];
}

export interface PredictionPayload {
  patient_id: string;
  modality: string;
  diseases?: string[];
  image_path: string;
  image_base64?: string;
  image_filename?: string;
}

export interface DiseaseCatalog {
  ocular: string[];
  neurological: string[];
  all: string[];
}
