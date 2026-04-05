import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { WorkspaceScopeSelector } from './WorkspaceScopeSelector';
import { renderWithQueryClient } from '../test/utils';

describe('WorkspaceScopeSelector', () => {
  it('shows an explicit error hint when the workspace list cannot be loaded', () => {
    renderWithQueryClient(
      <WorkspaceScopeSelector
        selectedScope="global"
        selectedWorkspaceId=""
        workspaceOptions={[]}
        workspaceOptionsPending={false}
        workspaceOptionsError={true}
        onSelectGlobal={() => {}}
        onSelectWorkspace={() => {}}
      />,
    );

    expect(screen.getByText('Unable to load the workspace list for this view.')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Workspace list unavailable')).toBeDisabled();
  });

  it('shows an explicit empty hint when no workspace-scoped data exists yet', () => {
    renderWithQueryClient(
      <WorkspaceScopeSelector
        selectedScope="global"
        selectedWorkspaceId=""
        workspaceOptions={[]}
        workspaceOptionsPending={false}
        workspaceOptionsError={false}
        onSelectGlobal={() => {}}
        onSelectWorkspace={() => {}}
      />,
    );

    expect(screen.getByText('No workspace-scoped data has landed for this view yet.')).toBeInTheDocument();
    expect(screen.getByDisplayValue('No workspaces yet')).toBeDisabled();
  });

  it('shows the active workspace in the hint and forwards selection changes', () => {
    const onSelectWorkspace = vi.fn();

    renderWithQueryClient(
      <WorkspaceScopeSelector
        selectedScope="workspace"
        selectedWorkspaceId="workspace-a"
        workspaceOptions={[{ key: 'workspace-a', label: 'Workspace A' }]}
        workspaceOptionsPending={false}
        workspaceOptionsError={false}
        onSelectGlobal={() => {}}
        onSelectWorkspace={onSelectWorkspace}
      />,
    );

    expect(screen.getByText('Scoped to Workspace A.')).toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue('Workspace A'), { target: { value: 'workspace-a' } });
    expect(onSelectWorkspace).toHaveBeenCalledWith('workspace-a');
  });
});
