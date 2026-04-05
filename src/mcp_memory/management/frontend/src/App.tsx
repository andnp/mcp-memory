import { lazy, Suspense } from 'react';
import { Link, NavLink, Route, Routes } from 'react-router-dom';

import { ActivityPage } from './pages/ActivityPage';
import { LogsPage } from './pages/LogsPage';
import { MemoryDetailPage } from './pages/MemoryDetailPage';
import { OverviewPage } from './pages/OverviewPage';
import { RetrievalPage } from './pages/RetrievalPage';
import { SearchPage } from './pages/SearchPage';

const NerdPage = lazy(async () => import('./pages/NerdPage').then((module) => ({ default: module.NerdPage })));
const SelectorStatsPage = lazy(async () => import('./pages/SelectorStatsPage').then((module) => ({ default: module.SelectorStatsPage })));

const navItems = [
  { to: '/', label: 'Overview', end: true },
  { to: '/search', label: 'Search' },
  { to: '/activity', label: 'Activity' },
  { to: '/logs', label: 'Logs' },
  { to: '/retrieval', label: 'Retrieval' },
  { to: '/nerd', label: 'Nerd' },
  { to: '/selector-stats', label: 'Selector Stats' },
];

function NotFoundPage() {
  return (
    <section className="panel p-4">
      <p className="panel-title">Not found</p>
      <h2 className="mt-1 text-lg font-semibold text-text">That dashboard page doesn&apos;t exist.</h2>
      <p className="mt-2 text-xs text-muted">
        The link may be stale, misspelled, or from an older build. Head back to the overview and we&apos;ll pretend this never happened.
      </p>
      <Link
        to="/"
        className="mt-4 inline-flex rounded-lg border border-accent bg-accent px-3 py-2 text-xs font-semibold text-ink"
      >
        Go to overview
      </Link>
    </section>
  );
}

export default function App() {
  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-4 px-4 py-4 lg:px-6 lg:py-5">
      <header>
        <nav className="flex flex-wrap gap-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `rounded-full border px-3 py-1.5 text-xs transition ${
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

      <Suspense fallback={<section className="panel p-3 text-xs text-muted">Loading page…</section>}>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/activity" element={<ActivityPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="/retrieval" element={<RetrievalPage />} />
          <Route path="/nerd" element={<NerdPage />} />
          <Route path="/selector-stats" element={<SelectorStatsPage />} />
          <Route path="/memory/:memoryId" element={<MemoryDetailPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </main>
  );
}