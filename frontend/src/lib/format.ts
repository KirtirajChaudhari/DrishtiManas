export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const num = (v: number) => v.toLocaleString("en-US");
export const fixed = (v: number, digits = 3) => v.toFixed(digits);

export const ACTIVATION_LABEL: Record<string, string> = {
  relu: "ReLU",
  leaky_relu: "Leaky ReLU",
  sigmoid: "Sigmoid",
  tanh: "Tanh",
  linear: "Linear"
};

export const OPTIMIZER_LABEL: Record<string, string> = {
  sgd: "SGD",
  momentum: "SGD + momentum",
  adam: "Adam"
};

export const LOSS_LABEL: Record<string, string> = {
  cross_entropy: "Cross-entropy",
  mse: "Mean squared error"
};
