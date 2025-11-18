import PredictionCard from "../components/PredictionCard";
import SummaryStats from "../components/SummaryStats";
import { useDoctorCases } from "../hooks/useDoctorCases";

export default function DoctorDashboard() {
  const { cases, metrics } = useDoctorCases();

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Clinician worklist</h2>
        <p className="text-sm text-slate-500">
          Layout mirrors the Ocular Disease Identifier research UI—triaged cases arrive sorted by risk so you
          can validate Grad-CAM heatmaps quickly.
        </p>
      </div>

      <SummaryStats {...metrics} />

      <section className="space-y-4">
        {cases.map((caseItem) => (
          <div key={caseItem.id} className="rounded-3xl border border-slate-100 bg-white p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Patient</p>
                <p className="font-semibold text-slate-900">{caseItem.patient_id}</p>
              </div>
              <div className="text-sm text-slate-500">
                {new Date(caseItem.created_at).toLocaleString()}
              </div>
              <button className="rounded-full border border-slate-200 px-4 py-1 text-sm font-medium text-slate-600">
                Mark reviewed
              </button>
            </div>
            <div className="mt-4 space-y-3">
              {caseItem.results.map((result) => (
                <PredictionCard key={result.disease} result={result} />
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
