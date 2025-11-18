import { useState } from "react";

import { createPrediction } from "../lib/api";
import type { PredictionPayload, PredictionResponse } from "../types/prediction";

export function usePredictions() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PredictionResponse | null>(null);

  const submit = async (payload: PredictionPayload) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await createPrediction(payload);
      setData(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return {
    data,
    isLoading,
    error,
    submit
  };
}
