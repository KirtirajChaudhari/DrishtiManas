import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { useDarkMode } from "../hooks/useDarkMode";
import type { EpochStats, TuningRun } from "../lib/api";
import { pct } from "../lib/format";

// Validated categorical palette (fixed order) — light and dark steps.
const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

function useTheme() {
  const dark = useDarkMode();
  return {
    series: dark ? SERIES_DARK : SERIES_LIGHT,
    grid: dark ? "#2e2d2a" : "#e7e5e4",
    axis: dark ? "#a8a29e" : "#78716c",
    tooltipBg: dark ? "#1c1917" : "#ffffff",
    tooltipBorder: dark ? "#44403c" : "#e7e5e4",
    text: dark ? "#f5f5f4" : "#1c1917"
  };
}

function useCommon() {
  const t = useTheme();
  return {
    t,
    axisProps: { stroke: t.axis, tick: { fill: t.axis, fontSize: 12 }, tickLine: false, axisLine: { stroke: t.grid } },
    tooltipProps: {
      contentStyle: {
        background: t.tooltipBg,
        border: `1px solid ${t.tooltipBorder}`,
        borderRadius: 10,
        fontSize: 12,
        color: t.text
      },
      labelStyle: { color: t.text, fontWeight: 600 },
      cursor: { stroke: t.axis, strokeDasharray: "3 3" }
    }
  };
}

export function TrainingCurve({
  history,
  metric,
  height = 260
}: {
  history: EpochStats[];
  metric: "loss" | "acc";
  height?: number;
}) {
  const { t, axisProps, tooltipProps } = useCommon();
  const fmt = (v: number) => (metric === "acc" ? pct(v) : v.toFixed(3));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={history} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="epoch" {...axisProps} label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: t.axis, fontSize: 12 }} height={36} />
        <YAxis {...axisProps} width={52} tickFormatter={fmt} domain={metric === "acc" ? ["auto", "auto"] : [0, "auto"]} />
        <Tooltip {...tooltipProps} formatter={(v: number) => fmt(v)} labelFormatter={(e) => `Epoch ${e}`} />
        <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12, color: t.axis }} />
        <Line type="monotone" dataKey={`train_${metric}`} name="Training" stroke={t.series[0]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
        <Line type="monotone" dataKey={`val_${metric}`} name="Validation" stroke={t.series[1]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Validation-loss curves of every run in one tuning experiment. */
export function RunsLossChart({ runs, height = 260 }: { runs: TuningRun[]; height?: number }) {
  const { t, axisProps, tooltipProps } = useCommon();
  const maxEpochs = Math.max(...runs.map((r) => r.history?.length ?? 0));
  const data = Array.from({ length: maxEpochs }, (_, i) => {
    const row: Record<string, number> = { epoch: i + 1 };
    runs.forEach((r) => {
      const h = r.history?.[i];
      if (h) row[r.label] = h.val_loss;
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="epoch" {...axisProps} height={36} label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: t.axis, fontSize: 12 }} />
        <YAxis {...axisProps} width={52} tickFormatter={(v: number) => v.toFixed(2)} domain={[0, "auto"]} />
        <Tooltip {...tooltipProps} formatter={(v: number) => v.toFixed(4)} labelFormatter={(e) => `Epoch ${e}`} />
        <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
        {runs.map((r, i) => (
          <Line
            key={r.label}
            type="monotone"
            dataKey={r.label}
            stroke={t.series[i % t.series.length]}
            strokeWidth={2}
            dot={false}
            connectNulls
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function RunsAccuracyBars({ runs, height = 260 }: { runs: TuningRun[]; height?: number }) {
  const { t, axisProps, tooltipProps } = useCommon();
  const min = Math.min(...runs.flatMap((r) => [r.val_acc, r.train_acc]));
  const floor = Math.max(0, Math.floor((min - 0.05) * 10) / 10);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={runs} margin={{ top: 8, right: 12, bottom: 4, left: 0 }} barGap={2}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="label" {...axisProps} interval={0} height={36} />
        <YAxis {...axisProps} width={52} domain={[floor, 1]} tickFormatter={(v: number) => pct(v, 0)} />
        <Tooltip {...tooltipProps} cursor={{ fill: t.grid, opacity: 0.4 }} formatter={(v: number) => pct(v, 2)} />
        <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="train_acc" name="Training accuracy" fill={t.series[0]} radius={[4, 4, 0, 0]} maxBarSize={36} />
        <Bar dataKey="val_acc" name="Validation accuracy" fill={t.series[1]} radius={[4, 4, 0, 0]} maxBarSize={36} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function FunctionPlot({
  fn,
  deriv,
  height = 180
}: {
  fn: (z: number) => number;
  deriv: (z: number) => number;
  height?: number;
}) {
  const { t, axisProps, tooltipProps } = useCommon();
  const data = Array.from({ length: 81 }, (_, i) => {
    const z = -4 + i * 0.1;
    return { z: Number(z.toFixed(1)), g: fn(z), d: deriv(z) };
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="z" type="number" domain={[-4, 4]} ticks={[-4, -2, 0, 2, 4]} {...axisProps} />
        <YAxis {...axisProps} width={44} tickFormatter={(v: number) => v.toFixed(1)} />
        <Tooltip {...tooltipProps} formatter={(v: number) => v.toFixed(3)} labelFormatter={(z) => `z = ${z}`} />
        <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="g" name="g(z)" stroke={t.series[0]} strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="d" name="g′(z)" stroke={t.series[1]} strokeWidth={2} strokeDasharray="5 4" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Confusion matrix as an HTML grid: sequential blue shading, count + row percentage in every cell. */
export function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const dark = useDarkMode();
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));
  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-separate border-spacing-1 text-sm" aria-label="Confusion matrix">
        <thead>
          <tr>
            <th className="p-1" />
            <th colSpan={labels.length} className="pb-1 text-center text-xs font-medium muted">
              Predicted class
            </th>
          </tr>
          <tr>
            <th className="p-1 text-right text-xs font-medium muted">True class</th>
            {labels.map((l) => (
              <th key={l} scope="col" className="min-w-[64px] px-1 text-center text-xs font-semibold">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th scope="row" className="pr-2 text-right text-xs font-semibold">
                {labels[i]}
              </th>
              {row.map((v, j) => {
                const share = rowTotals[i] ? v / rowTotals[i] : 0;
                // Blue ramp 100 -> 650 (OKLCH-ish steps) interpolated by row share.
                const light = `hsl(213 ${60 + share * 10}% ${95 - share * 55}%)`;
                const darkBg = `hsl(213 ${45 + share * 25}% ${14 + share * 38}%)`;
                const strong = share > 0.45;
                return (
                  <td
                    key={j}
                    title={`True ${labels[i]} → predicted ${labels[j]}: ${v} (${(share * 100).toFixed(1)}% of ${labels[i]})`}
                    className="h-16 min-w-[64px] rounded-lg text-center align-middle"
                    style={{
                      background: dark ? darkBg : light,
                      color: dark ? "#fafaf9" : strong ? "#ffffff" : "#1c1917",
                      outline: i === j ? `2px solid ${dark ? "#e7e5e4" : "#1c1917"}` : undefined,
                      outlineOffset: -2
                    }}
                  >
                    <span className="block text-base font-semibold tabular-nums">{v}</span>
                    <span className="block text-[11px] tabular-nums opacity-80">{(share * 100).toFixed(0)}%</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-center text-xs muted">
        Outlined diagonal = correct predictions. Percentages are shares of each true class (recall).
      </p>
    </div>
  );
}
