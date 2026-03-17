import { NavLink, Route, Routes } from 'react-router-dom';

import { ActivityPage } from './pages/ActivityPage';
import { LogsPage } from './pages/LogsPage';
import { MemoryDetailPage } from './pages/MemoryDetailPage';
import { NerdPage } from './pages/NerdPage';
import { OverviewPage } from './pages/OverviewPage';
import { SearchPage } from './pages/SearchPage';

const navItems = [
  { to: '/', label: 'Overview', end: true },
  { to: '/search', label: 'Search' },
  { to: '/activity', label: 'Activity' },
  { to: '/logs', label: 'Logs' },
  { to: '/nerd', label: 'Nerd' },
];

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

      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/activity" element={<ActivityPage />} />
        <Route path="/logs" element={<LogsPage />} />
        <Route path="/nerd" element={<NerdPage />} />
        <Route path="/memory/:memoryId" element={<MemoryDetailPage />} />
      </Routes>
    </main>
  );
}