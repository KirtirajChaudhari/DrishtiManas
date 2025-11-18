import { useEffect, useState } from "react";

import PredictionCard from "../components/PredictionCard";
import UploadForm from "../components/UploadForm";
import { usePredictions } from "../hooks/usePredictions";
import { pingHealth } from "../lib/api";
import type { PredictionPayload } from "../types/prediction";

export default function TechnicianUpload() {
  const { data, error, isLoading, submit } = usePredictions();
  const [health, setHealth] = useState<"unknown" | "online" | "offline">("unknown");
  const [healthMessage, setHealthMessage] = useState<string>("Checking FastAPI service...");

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        await pingHealth();
        if (isMounted) {
          setHealth("online");
          setHealthMessage("FastAPI backend reachable");
        }
      } catch (err) {
        if (isMounted) {
          setHealth("offline");
          setHealthMessage(
            err instanceof Error ? err.message : "Unable to reach FastAPI. Check docker-compose up."
          );
        }
      }
    };
    checkHealth();
    const intervalId = window.setInterval(checkHealth, 30000);
    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const handleSubmit = async (payload: PredictionPayload) => {
    await submit(payload);
  };

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Upload imaging study</h2>
        <p className="mb-4 text-sm text-slate-500">
          Borrowing from the Dry Eye and fundus CNN references, double-check tear film quality, centered
          optic disc, and pediatric comfort cues before running AI triage.
        </p>
        <UploadForm onSubmit={handleSubmit} />
      </div>

      <div className="space-y-4">
        <div className="rounded-3xl bg-white p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400">AI status</p>
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                health === "online"
                  ? "bg-emerald-50 text-emerald-700"
                  : health === "offline"
                    ? "bg-red-50 text-red-600"
                    : "bg-slate-100 text-slate-500"
              }`}
            >
              {health === "online" ? "Online" : health === "offline" ? "Offline" : "Pending"}
            </span>
          </div>
          <p className="pt-2 text-xs text-slate-500">{healthMessage}</p>
          {isLoading && <p className="pt-2 text-slate-600">Running inference...</p>}
          {error && <p className="pt-2 text-sm text-red-500">{error}</p>}
          {!isLoading && !error && !data && (
            <p className="pt-2 text-sm text-slate-500">Run a study to preview explainable results.</p>
          )}
        </div>

        {data && (
          <div className="space-y-3">
            <p className="text-sm font-semibold text-slate-600">
              Patient {data.patient_id} • {new Date(data.created_at).toLocaleString()}
            </p>
            <div className="space-y-3">
              {data.results.map((result) => (
                <PredictionCard key={result.disease} result={result} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
