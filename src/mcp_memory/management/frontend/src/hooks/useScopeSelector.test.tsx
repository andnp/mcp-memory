import { screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { useScopeSelector } from './useScopeSelector';
import { renderWithQueryClient } from '../test/utils';

type Option = { key: string; label: string };

function Harness({ options }: { options: Option[] }) {
  const scopeSelector = useScopeSelector<{ options: Option[] }, Option>({
    queryKey: ['harness', options.map((option) => option.key).join(',')],
    fetchOptions: async () => ({ options }),
    deriveOptions: (data) => data?.options ?? [],
  });

  return (
    <div>
      <span data-testid="scope">{scopeSelector.selectedScope}</span>
      <span data-testid="workspace-id">{scopeSelector.selectedWorkspaceId}</span>
      <span data-testid="pending">{String(scopeSelector.workspaceOptionsPending)}</span>
      <button type="button" onClick={scopeSelector.selectGlobal}>
        global
      </button>
      <button type="button" onClick={() => scopeSelector.selectWorkspace('a')}>
        select-a
      </button>
      <button type="button" onClick={() => scopeSelector.selectWorkspace('b')}>
        select-b
      </button>
      <button type="button" onClick={() => scopeSelector.selectWorkspace('')}>
        select-empty
      </button>
    </div>
  );
}

async function waitForOptionsLoaded() {
  await waitFor(() => expect(screen.getByTestId('pending')).toHaveTextContent('false'));
}

describe('useScopeSelector', () => {
  it('starts scoped to global with no workspace selected', () => {
    renderWithQueryClient(<Harness options={[{ key: 'a', label: 'A' }]} />);

    expect(screen.getByTestId('scope')).toHaveTextContent('global');
    expect(screen.getByTestId('workspace-id')).toHaveTextContent('');
  });

  it('switches to the workspace scope when a workspace is selected', async () => {
    renderWithQueryClient(<Harness options={[{ key: 'a', label: 'A' }]} />);
    await waitForOptionsLoaded();

    fireEvent.click(screen.getByText('select-a'));

    expect(screen.getByTestId('scope')).toHaveTextContent('workspace');
    expect(screen.getByTestId('workspace-id')).toHaveTextContent('a');
  });

  it('falls back to another option when the selected workspace disappears', async () => {
    renderWithQueryClient(<Harness options={[{ key: 'a', label: 'A' }]} />);
    await waitForOptionsLoaded();

    fireEvent.click(screen.getByText('select-b'));

    await waitFor(() => expect(screen.getByTestId('workspace-id')).toHaveTextContent('a'));
    expect(screen.getByTestId('scope')).toHaveTextContent('workspace');
  });

  it('resets to global when no workspace options remain', async () => {
    renderWithQueryClient(<Harness options={[]} />);
    await waitForOptionsLoaded();

    fireEvent.click(screen.getByText('select-b'));

    await waitFor(() => expect(screen.getByTestId('scope')).toHaveTextContent('global'));
  });

  it('resets to global when selecting an empty workspace id', async () => {
    renderWithQueryClient(<Harness options={[{ key: 'a', label: 'A' }]} />);
    await waitForOptionsLoaded();

    fireEvent.click(screen.getByText('select-a'));
    fireEvent.click(screen.getByText('select-empty'));

    expect(screen.getByTestId('scope')).toHaveTextContent('global');
    expect(screen.getByTestId('workspace-id')).toHaveTextContent('');
  });
});
