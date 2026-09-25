import clsx from "clsx";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { api } from "./lib/api";

const NAV = [
  { to: "/", label: "Classify", end: true },
  { to: "/model", label: "Model report", end: false },
  { to: "/learn", label: "How it works", end: false }
];

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-8 w-8 shrink-0" aria-hidden="true">
      <rect width="32" height="32" rx="8" className="fill-brand-700" />
      <path d="M4 16c3-5.5 7.3-8 12-8s9 2.5 12 8c-3 5.5-7.3 8-12 8S7 21.5 4 16z" fill="none" stroke="white" strokeWidth="2" />
      <circle cx="16" cy="16" r="4.2" fill="white" />
    </svg>
  );
}

function ApiStatus() {
  const [state, setState] = useState<"checking" | "online" | "offline">("checking");
  useEffect(() => {
    api
      .health()
      .then((h) => setState(h.model.loaded ? "online" : "offline"))
      .catch(() => setState("offline"));
  }, []);
  return (
    <span className="inline-flex items-center gap-2 text-xs muted" role="status">
      <span
        className={clsx(
          "h-2 w-2 rounded-full",
          state === "online" && "bg-emerald-500",
          state === "offline" && "bg-red-500",
          state === "checking" && "bg-stone-400"
        )}
      />
      {state === "online" ? "Model online" : state === "offline" ? "Model offline" : "Connecting…"}
    </span>
  );
}

export default function App() {
  const { pathname } = useLocation();
  useEffect(() => window.scrollTo(0, 0), [pathname]);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-stone-200 bg-stone-50/90 backdrop-blur dark:border-stone-800 dark:bg-stone-950/90">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
          <NavLink to="/" className="flex items-center gap-3">
            <Logo />
            <span className="leading-tight">
              <span className="block text-[15px] font-semibold tracking-tight">DrishtiManas</span>
              <span className="block text-xs muted">Retinal OCT classifier</span>
            </span>
          </NavLink>
          <nav className="order-3 -mx-1 flex w-full gap-1 overflow-x-auto sm:order-2 sm:w-auto" aria-label="Main">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    "whitespace-nowrap rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900"
                      : "text-stone-600 hover:bg-stone-200/70 hover:text-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-100"
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="order-2 sm:order-3">
            <ApiStatus />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
        <Outlet />
      </main>

      <footer className="border-t border-stone-200 dark:border-stone-800">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-xs muted sm:flex-row sm:justify-between sm:px-6">
          <p>
            Research and education only. Not a medical device; a qualified clinician must make any diagnosis.
          </p>
          <p>
            Data: OCTMNIST (MedMNIST v2, CC BY 4.0) · Kermany et al., Cell 2018
          </p>
        </div>
      </footer>
    </div>
  );
}
