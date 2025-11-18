import { NavLink, Outlet } from "react-router-dom";

function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-white shadow-sm">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-6 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.4em] text-primary-dark">Drishti Manas</p>
            <h1 className="text-2xl font-semibold text-primary">Smart ocular diagnostics</h1>
          </div>
          <nav className="flex gap-2">
            <NavLink
              to="/technician"
              className={({ isActive }) =>
                `rounded-full px-5 py-2 text-sm font-medium transition ${
                  isActive ? "bg-primary text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`
              }
            >
              Technician Workspace
            </NavLink>
            <NavLink
              to="/doctor"
              className={({ isActive }) =>
                `rounded-full px-5 py-2 text-sm font-medium transition ${
                  isActive ? "bg-primary text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`
              }
            >
              Doctor Dashboard
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-10">
        <Outlet />
      </main>
    </div>
  );
}

export default App;
