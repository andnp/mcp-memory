import type { OverviewResponse } from '../lib/api';

interface PulseOverviewProps {
  overview: OverviewResponse;
}

export function PulseOverview({ overview }: PulseOverviewProps) {
  const cards = [
    ['Memories', overview.memories.total],
    ['Memory Lines', overview.memory_metrics.total_memory_lines],
    ['Compressed Lines', overview.memory_metrics.total_lines_compressed],
    ['Thought Buffer', overview.memory_metrics.thought_buffer_entries],
    ['Failed Tasks', overview.tasks.failed_count],
    ['Embedding Backend', overview.embeddings.backend ?? 'none'],
    ['SQLite Size (bytes)', overview.storage.sqlite_bytes],
  ];

  return (
    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {cards.map(([label, value]) => (
        <article key={label} className="metric-card">
          <p className="panel-title">{label}</p>
          <p className="mt-3 text-3xl font-semibold text-text">{value}</p>
        </article>
      ))}
    </section>
  );
}