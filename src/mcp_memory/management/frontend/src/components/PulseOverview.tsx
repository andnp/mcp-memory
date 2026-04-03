import type { OverviewResponse } from '../lib/api';

interface PulseOverviewProps {
  overview: OverviewResponse;
}

export function PulseOverview({ overview }: PulseOverviewProps) {
  const formatPremiumUsage = (value: number) => {
    if (Number.isInteger(value)) {
      return value.toString();
    }
    return value.toFixed(2).replace(/\.00$/, '');
  };

  const cards = [
    ['Memories', overview.memories.total],
    ['Memory Lines', overview.memory_metrics.total_memory_lines],
    ['Compressed Lines', overview.memory_metrics.total_lines_compressed],
    ['Copilot Premium Today', overview.premium_usage.copilot_premium_requests_today],
    ['Thought Buffer', overview.memory_metrics.thought_buffer_entries],
    ['Failed Tasks', overview.tasks.failed_count],
    ['Embedding Backend', overview.embeddings.backend ?? 'none'],
    ['SQLite Size (bytes)', overview.storage.sqlite_bytes],
  ];

  return (
    <section className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">
      {cards.map(([label, value]) => (
        <article key={label} className="metric-card">
          <p className="panel-title">{label}</p>
          <p className="mt-1 text-xl font-semibold text-text">
            {label === 'Copilot Premium Today' ? formatPremiumUsage(value as number) : value}
          </p>
          {label === 'Copilot Premium Today' ? (
            <p className="mt-1 text-xs text-muted">
              {formatPremiumUsage(overview.premium_usage.copilot_premium_requests_last_day)} in the last 24h
            </p>
          ) : null}
        </article>
      ))}
    </section>
  );
}