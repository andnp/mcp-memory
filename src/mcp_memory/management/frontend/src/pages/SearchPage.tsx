import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Link } from 'react-router-dom';

import { searchMemories } from '../lib/api';

export function SearchPage() {
  const [query, setQuery] = useState('');
  const searchMutation = useMutation({
    mutationFn: searchMemories,
  });

  return (
    <section className="space-y-6">
      <section className="panel p-4">
        <p className="panel-title">Explore</p>
        <h2 className="mt-1 text-lg font-semibold text-text">Search memories</h2>
        <div className="mt-3 flex flex-col gap-2 lg:flex-row">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && query.trim()) {
                searchMutation.mutate({ query: query.trim(), limit: 12 });
              }
            }}
            className="w-full rounded-lg border border-border bg-ink px-3 py-2 text-xs text-text outline-none focus:border-accent"
            placeholder="Search by topic, person, memory title, or concept"
          />
          <button
            type="button"
            className="rounded-lg border border-accent bg-accent px-3 py-2 text-xs font-semibold text-ink"
            onClick={() => query.trim() && searchMutation.mutate({ query: query.trim(), limit: 12 })}
          >
            Search
          </button>
        </div>
      </section>

      <section className="table-shell">
        <div className="border-b border-border px-4 py-4">
          <p className="panel-title">Results</p>
          <h2 className="mt-2 text-lg font-semibold text-text">Summary-first memory search</h2>
        </div>
        <table>
          <thead>
            <tr>
              <th>Score</th>
              <th>Memory</th>
              <th>Type</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {searchMutation.data?.results.length ? searchMutation.data.results.map((result) => (
              <tr key={result.memory_id}>
                <td>{result.score.toFixed(3)}</td>
                <td>
                  <Link className="font-medium text-accent" to={`/memory/${result.memory_id}`}>{result.title}</Link>
                  {result.summary ? <div className="mt-1 text-xs text-muted">{result.summary}</div> : null}
                  {result.tags.length ? <div className="mt-2 text-xs text-muted">{result.tags.join(', ')}</div> : null}
                </td>
                <td>{result.memory_type}</td>
                <td>{result.status}</td>
              </tr>
            )) : (
              <tr>
                <td colSpan={4} className="text-muted">
                  {searchMutation.isPending ? 'Searching…' : 'Run a search to explore the memory graph.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </section>
  );
}