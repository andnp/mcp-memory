import type { OverviewResponse } from '../lib/api';

interface PulseOverviewProps {
  overview: OverviewResponse;
}

export function PulseOverview({ overview }: PulseOverviewProps) {
  const cards = [
    ['Memories', overview.memories.total],
    ['Memory Lines', overview.memory_metrics.total_memory_lines],
    ['Compressed Lines', overview.memory_metrics.total_lines_compressed],
    ['Provider Tokens (24h)', overview.token_usage.total_tokens_last_day],
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
            {value}
          </p>
          {label === 'Provider Tokens (24h)' ? (
            <p className="mt-1 text-xs text-muted">
              {overview.token_usage.provider_calls_last_day} provider calls · {overview.token_usage.output_tokens_last_day} output tokens
            </p>
          ) : null}
        </article>
      ))}
    </section>
  );
}
