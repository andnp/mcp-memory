import { useEffect, useMemo, useState } from 'react';
import { useQuery, type QueryKey } from '@tanstack/react-query';

import type { ManagementScope } from '../lib/api';

interface UseScopeSelectorOptions<TData, TOption extends { key: string }> {
  queryKey: QueryKey;
  fetchOptions: () => Promise<TData>;
  deriveOptions: (data: TData | undefined) => TOption[];
}

interface UseScopeSelectorResult<TOption extends { key: string }> {
  selectedScope: ManagementScope;
  selectedWorkspaceId: string;
  workspaceOptions: TOption[];
  workspaceOptionsPending: boolean;
  workspaceOptionsError: boolean;
  selectGlobal: () => void;
  selectWorkspace: (workspaceId: string) => void;
}

/**
 * Shared "global vs scoped-entity" selection state: tracks the selected scope,
 * fetches the entity options for the scope picker, and auto-falls back to
 * "global" when the currently-selected entity disappears from the options.
 */
export function useScopeSelector<TData, TOption extends { key: string }>({
  queryKey,
  fetchOptions,
  deriveOptions,
}: UseScopeSelectorOptions<TData, TOption>): UseScopeSelectorResult<TOption> {
  const [selectedScope, setSelectedScope] = useState<ManagementScope>('global');
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState('');

  const workspaceOptionsQuery = useQuery({
    queryKey,
    queryFn: fetchOptions,
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const workspaceOptions = useMemo(
    () => deriveOptions(workspaceOptionsQuery.data),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [workspaceOptionsQuery.data],
  );

  useEffect(() => {
    if (selectedScope !== 'workspace') {
      return;
    }
    if (workspaceOptions.some((option) => option.key === selectedWorkspaceId)) {
      return;
    }
    const fallbackWorkspaceId = workspaceOptions[0]?.key ?? '';
    if (fallbackWorkspaceId) {
      setSelectedWorkspaceId(fallbackWorkspaceId);
      return;
    }
    setSelectedScope('global');
  }, [selectedScope, selectedWorkspaceId, workspaceOptions]);

  function selectGlobal() {
    setSelectedScope('global');
  }

  function selectWorkspace(workspaceId: string) {
    if (!workspaceId) {
      setSelectedScope('global');
      setSelectedWorkspaceId('');
      return;
    }
    setSelectedWorkspaceId(workspaceId);
    setSelectedScope('workspace');
  }

  return {
    selectedScope,
    selectedWorkspaceId,
    workspaceOptions,
    workspaceOptionsPending: workspaceOptionsQuery.isPending,
    workspaceOptionsError: workspaceOptionsQuery.isError,
    selectGlobal,
    selectWorkspace,
  };
}
