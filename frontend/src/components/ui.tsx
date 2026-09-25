import clsx from "clsx";
import type { ReactNode } from "react";

import type { Urgency } from "../lib/api";

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-8 max-w-3xl">
      <p className="eyebrow">{eyebrow}</p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
      {children && <div className="mt-3 text-[15px] leading-relaxed muted">{children}</div>}
    </div>
  );
}

export function Section({
  id,
  title,
  subtitle,
  children,
  className
}: {
  id?: string;
  title: string;
  subtitle?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={clsx("scroll-mt-24", className)}>
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      {subtitle && <p className="mt-1 max-w-3xl text-sm muted">{subtitle}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="card !p-5">
      <p className="text-sm muted">{label}</p>
      <p className="mt-1 text-3xl font-semibold tabular-nums tracking-tight">{value}</p>
      {hint && <p className="mt-1 text-xs muted">{hint}</p>}
    </div>
  );
}

const URGENCY: Record<Urgency, { label: string; className: string; icon: string }> = {
  urgent: {
    label: "Refer urgently",
    className: "bg-red-50 text-red-800 ring-red-200 dark:bg-red-950/60 dark:text-red-200 dark:ring-red-900",
    icon: "!"
  },
  routine: {
    label: "Routine follow-up",
    className: "bg-amber-50 text-amber-900 ring-amber-200 dark:bg-amber-950/60 dark:text-amber-200 dark:ring-amber-900",
    icon: "•"
  },
  none: {
    label: "No referral",
    className: "bg-emerald-50 text-emerald-800 ring-emerald-200 dark:bg-emerald-950/60 dark:text-emerald-200 dark:ring-emerald-900",
    icon: "✓"
  }
};

export function UrgencyBadge({ urgency }: { urgency: Urgency }) {
  const u = URGENCY[urgency];
  return (
    <span className={clsx("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1", u.className)}>
      <span aria-hidden="true" className="font-bold">
        {u.icon}
      </span>
      {u.label}
    </span>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm muted" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-stone-300 border-t-brand-600" />
      {label}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200">
      {message}
    </div>
  );
}

export function Table({ head, rows, align }: { head: ReactNode[]; rows: ReactNode[][]; align?: ("left" | "right")[] }) {
  return (
    <div className="-mx-1 overflow-x-auto">
      <table className="w-full min-w-[480px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-stone-200 text-left dark:border-stone-700">
            {head.map((h, i) => (
              <th
                key={i}
                scope="col"
                className={clsx("px-2 py-2 font-medium muted", align?.[i] === "right" && "text-right")}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} className="border-b border-stone-100 last:border-0 dark:border-stone-800">
              {row.map((cell, c) => (
                <td key={c} className={clsx("px-2 py-2", align?.[c] === "right" && "text-right tabular-nums")}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
