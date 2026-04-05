import type { NerdMetricsScope } from '../lib/api';

type WorkspaceOption = {
  key: string;
  label: string;
};

function selectorPlaceholder(
  isPending: boolean,
  isError: boolean,
  workspaceOptions: WorkspaceOption[],
): string {
  if (isPending) {
    return 'Loading workspaces…';
  }
  if (isError) {
    return 'Workspace list unavailable';
  }
  if (!workspaceOptions.length) {
    return 'No workspaces yet';
  }
  return 'Workspace…';
}

function selectorHint(
  isPending: boolean,
  isError: boolean,
  workspaceOptions: WorkspaceOption[],
  selectedScope: NerdMetricsScope,
  selectedWorkspaceId: string,
): string {
  if (isPending) {
    return 'Loading workspace-scoped options…';
  }
  if (isError) {
    return 'Unable to load the workspace list for this view.';
  }
  if (!workspaceOptions.length) {
    return 'No workspace-scoped data has landed for this view yet.';
  }
  if (selectedScope === 'workspace') {
    const selectedOption = workspaceOptions.find((option) => option.key === selectedWorkspaceId);
    if (selectedOption) {
      return `Scoped to ${selectedOption.label}.`;
    }
  }
  return 'Global by default; switch to a workspace when you want a narrower slice.';
}

export function WorkspaceScopeSelector({
  selectedScope,
  selectedWorkspaceId,
  workspaceOptions,
  workspaceOptionsPending,
  workspaceOptionsError,
  onSelectGlobal,
  onSelectWorkspace,
}: {
  selectedScope: NerdMetricsScope;
  selectedWorkspaceId: string;
  workspaceOptions: WorkspaceOption[];
  workspaceOptionsPending: boolean;
  workspaceOptionsError: boolean;
  onSelectGlobal: () => void;
  onSelectWorkspace: (workspaceId: string) => void;
}) {
  const disabled = workspaceOptionsPending || workspaceOptionsError || workspaceOptions.length === 0;
  const placeholder = selectorPlaceholder(workspaceOptionsPending, workspaceOptionsError, workspaceOptions);
  const hint = selectorHint(
    workspaceOptionsPending,
    workspaceOptionsError,
    workspaceOptions,
    selectedScope,
    selectedWorkspaceId,
  );

  return (
    <section className="panel flex flex-col gap-2 p-1">
      <div className="flex items-center gap-2">
        <span className="px-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">Scope</span>
        <button
          type="button"
          onClick={onSelectGlobal}
          className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${selectedScope === 'global'
            ? 'border-accent bg-accent text-ink'
            : 'border-border bg-transparent text-muted hover:border-accent hover:text-text'
          }`}
        >
          Global
        </button>
        <label className="flex items-center">
          <span className="sr-only">Workspace scope</span>
          <select
            value={selectedScope === 'workspace' ? selectedWorkspaceId : ''}
            onChange={(event) => onSelectWorkspace(event.target.value)}
            disabled={disabled}
            className="rounded-full border border-border bg-transparent px-3 py-1 text-[11px] font-medium text-text outline-none transition hover:border-accent disabled:cursor-not-allowed disabled:text-muted"
          >
            <option value="">{placeholder}</option>
            {workspaceOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="px-2 text-[11px] text-muted">{hint}</p>
    </section>
  );
}
