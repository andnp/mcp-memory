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
    <section className="panel p-6">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="panel-title">Command Bar</p>
          <h2 className="mt-2 text-xl font-semibold text-text">Stash thoughts and steer agents from one place</h2>
        </div>
        <span className="rounded-full border border-border px-3 py-1 text-xs text-muted">Ctrl+K</span>
      </div>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
        <label className="flex-1 text-sm text-muted">
          Input
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
            className="mt-2 w-full rounded-xl border border-border bg-ink px-4 py-3 text-sm text-text outline-none transition focus:border-accent"
            placeholder="Type a thought, or /run-agent deduplicator"
          />
        </label>
        <button
          type="button"
          disabled={submitting}
          onClick={() => void handleSubmit()}
          className="rounded-xl border border-accent bg-accent px-5 py-3 text-sm font-semibold text-ink transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting ? 'Working…' : 'Submit'}
        </button>
      </div>
      <p className="mt-3 text-sm text-muted">{hint}</p>
    </section>
  );
}