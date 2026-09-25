export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export type Urgency = "urgent" | "routine" | "none";

export interface ClassInfo {
  id: number;
  code: string;
  name: string;
  description: string;
  urgency: Urgency;
}

export interface ClassProbability extends ClassInfo {
  probability: number;
}

export interface PredictionResponse {
  filename: string | null;
  image_size: [number, number];
  prediction: ClassProbability;
  probabilities: ClassProbability[];
  input_pixels: number[][];
  warnings: string[];
  latency_ms: number;
}

export interface Sample {
  file: string;
  label: string;
  name: string;
  url: string;
}

export interface EpochStats {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  grad_norm: number;
  learning_rate: number;
  seconds: number;
}

export interface ClassMetrics {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EvalReport {
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  precision_weighted: number;
  recall_weighted: number;
  f1_weighted: number;
  per_class: ClassMetrics[];
  confusion_matrix: number[][];
}

export interface MLPConfig {
  input_dim: number;
  num_classes: number;
  hidden_layers: number[];
  activation: string;
  loss: string;
  optimizer: string;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  l2: number;
  lr_decay: number;
  early_stopping_patience: number | null;
  class_weights: number[] | null;
  dropout: number;
}

export interface TuningRun {
  value: number | string;
  label: string;
  val_acc: number;
  val_f1?: number;
  train_acc: number;
  val_loss?: number;
  train_loss?: number;
  epochs_run?: number;
  best_epoch?: number;
  seconds?: number;
  params?: number;
  history?: EpochStats[];
}

export interface SelectionRun {
  name: string;
  val_f1: number;
  val_acc: number | null;
  config: MLPConfig;
}

export interface TuningExperiment {
  key: string;
  title: string;
  incumbent?: number | string;
  best?: number | string;
  runs: TuningRun[] | SelectionRun[];
  history?: EpochStats[];
}

export interface AblationRow {
  step: string;
  val_acc: number;
  val_f1: number;
  test_acc: number;
  test_f1: number;
  test_per_class_f1: Record<string, number>;
}

export interface Baseline {
  name: string;
  description: string;
  val_acc: number;
  val_f1: number;
  test_acc: number;
  test_f1: number;
  seconds: number;
}

export interface ModelReport {
  generated_at: string;
  quick: boolean;
  dataset: {
    name: string;
    source: string;
    url: string;
    license: string;
    image_shape: number[];
    classes: ClassInfo[];
    counts: Record<"train" | "val" | "test", number[]>;
    tuning_subset_per_class: number;
    final_training_regime: string;
    final_training_size: number;
  };
  preprocessing: string[];
  features: {
    selected: string;
    spec: { kind: string; image_size: number; pixel_size: number; hog_cells: number[] };
    dim: number;
    comparison: (TuningRun & { dim: number })[];
  };
  baselines: Baseline[];
  gradient_check: { max_relative_error: number; passed: boolean };
  tuning: { base_config: MLPConfig; experiments: TuningExperiment[] };
  final: {
    config: MLPConfig;
    parameters: number;
    layers: number[];
    epochs_trained: number;
    best_epoch: number;
    training_seconds: number;
    history: EpochStats[];
    regimes: { regime: string; val_f1: number; val_acc: number; epochs: number }[];
    ensemble: {
      size: number;
      members: { seed: number; val_acc: number; val_f1: number }[];
      total_parameters: number;
      uncalibrated_validation: EvalReport;
    };
    calibration: {
      method: string;
      bias: number[];
      calibration_set_size?: number;
      val_f1_before: number;
      val_f1_after: number;
      full_validation_f1_before?: number;
      full_validation_f1_after?: number;
    };
    ablation?: {
      rows: AblationRow[];
      previous_version?: AblationRow;
      significance_vs_previous?: {
        test: string;
        overall: { prev_right_new_wrong: number; prev_wrong_new_right: number; p_value: number };
        drusen: {
          prev_right_new_wrong: number;
          prev_wrong_new_right: number;
          p_value: number;
          recall_before: number;
          recall_after: number;
        };
        accuracy_standard_error: number;
      };
      latency_ms_median?: number;
      latency_ms_p95?: number;
    };
    validation: EvalReport;
    test: EvalReport;
  };
  samples: Omit<Sample, "url">[];
  pipeline_seconds: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; version: string; model: { loaded: boolean } }>("/api/health"),
  report: () => request<ModelReport>("/api/model"),
  samples: () => request<Sample[]>("/api/samples"),
  predict: (file: Blob, filename = "upload.png") => {
    const form = new FormData();
    form.append("file", file, filename);
    return request<PredictionResponse>("/api/predict", { method: "POST", body: form });
  },
  assetUrl: (path: string) => `${API_BASE}${path}`
};
