import { isRouteErrorResponse, useRouteError } from "react-router-dom";

/**
 * React Router's route-level error boundary. Without this, any uncaught render
 * error (including transient third-party effect-timing glitches, e.g. a chart
 * library's animation cleanup racing a backgrounded-tab timer) replaces the
 * whole app with React Router's generic "Unexpected Application Error" screen.
 * This shows a recoverable message instead: reloading re-mounts everything
 * cleanly, since nothing here depends on persisted client state.
 */
export default function RouteError() {
  const error = useRouteError();
  const detail = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? error.message
      : "Something went wrong while rendering the page.";

  return (
    <div className="mx-auto flex min-h-screen max-w-lg flex-col items-center justify-center gap-4 px-6 text-center">
      <p className="eyebrow">Something went wrong</p>
      <h1 className="text-2xl font-semibold tracking-tight">This page hit an unexpected error</h1>
      <p className="text-sm muted">
        This is usually a transient glitch (for example, a browser tab that was in the background while a chart was
        loading), not a data problem. Reloading almost always fixes it.
      </p>
      <p className="rounded-lg bg-stone-100 px-3 py-2 font-mono text-xs text-stone-600 dark:bg-stone-800 dark:text-stone-400">
        {detail}
      </p>
      <button
        type="button"
        onClick={() => window.location.reload()}
        className="rounded-full bg-brand-700 px-5 py-2 text-sm font-medium text-white hover:bg-brand-800 dark:bg-brand-500 dark:text-stone-950 dark:hover:bg-brand-400"
      >
        Reload
      </button>
    </div>
  );
}
