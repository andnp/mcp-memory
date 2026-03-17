import { useEffect, useMemo, useRef, useState } from 'react';

import { recordThought, runAgent, runAllAgents, type CommandBarResult } from '../lib/api';

interface CommandBarProps {
  onResult: (text: string) => void;
}

function formatResult(result: CommandBarResult): string {
  return JSON.stringify(result, null, 2);
}

export function CommandBar({ onResult }: CommandBarProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [value, setValue] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, []);

  const hint = useMemo(
    () => 'Type a thought to stash it, or use /run-agent <name> and /run-all.',
    [],
  );

  async function handleSubmit() {
    const trimmed = value.trim();
    if (!trimmed) {
      onResult('Enter a thought or slash command.');
      return;
    }

    setSubmitting(true);
    try {
      if (trimmed.startsWith('/run-agent ')) {
        const taskName = trimmed.slice('/run-agent '.length).trim();
        onResult(formatResult(await runAgent(taskName)));
      } else if (trimmed === '/run-all' || trimmed === '/run-all-agents') {
        onResult(formatResult(await runAllAgents()));
      } else {
        onResult(formatResult(await recordThought(trimmed)));
      }
      setValue('');
    } catch (error) {
      onResult(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="panel p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <p className="panel-title">Command Bar</p>
          <p className="mt-1 text-xs text-muted">Stash a thought or run <code>/run-agent</code>.</p>
        </div>
        <span className="rounded-full border border-border px-2 py-0.5 text-[10px] text-muted">Ctrl+K</span>
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <label className="flex-1 text-xs text-muted">
          <input
            ref={inputRef}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                void handleSubmit();
              }
            }}
            className="mt-1 w-full rounded-lg border border-border bg-ink px-2.5 py-2 text-xs text-text outline-none transition focus:border-accent"
            placeholder="Type a thought, or /run-agent deduplicator"
          />
        </label>
        <button
          type="button"
          disabled={submitting}
          onClick={() => void handleSubmit()}
          className="rounded-lg border border-accent bg-accent px-3 py-2 text-xs font-semibold text-ink transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting ? 'Working…' : 'Submit'}
        </button>
      </div>
      <p className="mt-1.5 text-[11px] text-muted">{hint}</p>
    </section>
  );
}