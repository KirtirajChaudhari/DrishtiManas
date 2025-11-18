import { Activity, AlertTriangle, CheckCircle2, Timer } from "lucide-react";
import type { ReactElement } from "react";

import type { PredictionResult } from "../types/prediction";

const TRIAGE_STYLES: Record<PredictionResult["triage"], { label: string; color: string; icon: ReactElement }> = {
  critical: {
    label: "Critical",
    color: "text-red-600",
    icon: <AlertTriangle className="size-4 text-red-600" />
  },
  elevated: {
    label: "Elevated",
    color: "text-amber-600",
    icon: <Activity className="size-4 text-amber-600" />
  },
  watch: {
    label: "Watch",
    color: "text-blue-600",
    icon: <Timer className="size-4 text-blue-600" />
  },
  baseline: {
    label: "Baseline",
    color: "text-emerald-600",
    icon: <CheckCircle2 className="size-4 text-emerald-600" />
  }
};

interface PredictionCardProps {
  result: PredictionResult;
}

export default function PredictionCard({ result }: PredictionCardProps) {
  const triage = TRIAGE_STYLES[result.triage];

  return (
    <article className="rounded-3xl border border-slate-100 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Disease</p>
          <h3 className="text-lg font-semibold text-slate-900">{result.disease}</h3>
        </div>
        <div className={`flex items-center gap-2 text-sm font-semibold ${triage.color}`}>
          {triage.icon}
          {triage.label}
        </div>
      </div>

      <div className="mt-4">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Probability</p>
        <p className="text-3xl font-semibold text-slate-900">{(result.probability * 100).toFixed(1)}%</p>
        <div className="mt-2 h-2 rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${Math.min(result.probability * 100, 100)}%` }}
          />
        </div>
      </div>

      {result.heatmap?.url && (
        <figure className="mt-4 flex items-center gap-4">
          <img
            src={result.heatmap.url}
            alt={result.heatmap.caption ?? "Heatmap"}
            className="h-24 w-24 rounded-2xl object-cover"
          />
          <figcaption className="text-sm text-slate-500">{result.heatmap.caption}</figcaption>
        </figure>
      )}
    </article>
  );
}
