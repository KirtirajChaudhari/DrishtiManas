interface SummaryStatsProps {
  totalCases: number;
  criticalCases: number;
  sla: string;
}

export default function SummaryStats({ totalCases, criticalCases, sla }: SummaryStatsProps) {
  return (
    <section className="grid gap-4 md:grid-cols-3">
      <div className="rounded-3xl bg-white p-4 shadow-sm">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Cases today</p>
        <p className="text-3xl font-semibold text-slate-900">{totalCases}</p>
      </div>
      <div className="rounded-3xl bg-white p-4 shadow-sm">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Critical alerts</p>
        <p className="text-3xl font-semibold text-red-500">{criticalCases}</p>
      </div>
      <div className="rounded-3xl bg-white p-4 shadow-sm">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Median turnaround</p>
        <p className="text-3xl font-semibold text-slate-900">{sla}</p>
      </div>
    </section>
  );
}
