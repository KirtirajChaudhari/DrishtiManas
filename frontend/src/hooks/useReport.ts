import { useEffect, useState } from "react";

import { api, type ModelReport } from "../lib/api";

let cache: Promise<ModelReport> | null = null;

/** Fetches the training report once and shares it between pages. */
export function useReport() {
  const [report, setReport] = useState<ModelReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    cache ??= api.report().catch((err) => {
      cache = null;
      throw err;
    });
    cache.then(
      (r) => alive && setReport(r),
      (err: Error) => alive && setError(err.message)
    );
    return () => {
      alive = false;
    };
  }, []);

  return { report, error };
}
