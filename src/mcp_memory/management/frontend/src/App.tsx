import { NavLink, Route, Routes } from 'react-router-dom';

import { ActivityPage } from './pages/ActivityPage';
import { LogsPage } from './pages/LogsPage';
import { MemoryDetailPage } from './pages/MemoryDetailPage';
import { OverviewPage } from './pages/OverviewPage';
import { SearchPage } from './pages/SearchPage';

const navItems = [
  { to: '/', label: 'Overview', end: true },
  { to: '/search', label: 'Search' },
  { to: '/activity', label: 'Activity' },
  { to: '/logs', label: 'Logs' },
];

export default function App() {
  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-6 py-8 lg:px-10">
      <header className="flex flex-col gap-4">
        <div>
          <p className="panel-title">Memory Command Center</p>
          <h1 className="mt-2 text-4xl font-semibold tracking-tight text-text">Operator pages for plan 6</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
            The dashboard is now split into focused pages so search, audit activity, logs, and memory drill-down can grow without
            collapsing into one very expensive div.
          </p>
        </div>
        <nav className="flex flex-wrap gap-3">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `rounded-full border px-4 py-2 text-sm transition ${
                  isActive
                    ? 'border-accent bg-accent text-ink'
                    : 'border-border bg-panel/90 text-muted hover:border-accent hover:text-text'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/activity" element={<ActivityPage />} />
        <Route path="/logs" element={<LogsPage />} />
        <Route path="/memory/:memoryId" element={<MemoryDetailPage />} />
      </Routes>
    </main>
  );
}