import { fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ActivityPage } from './ActivityPage';
import { renderWithQueryClient } from '../test/utils';

function createJsonResponse(payload: unknown): Response {
  return {
    ok: true,
    json: async () => payload,
  } as Response;
}

function getConversationRequests(fetchMock: ReturnType<typeof vi.fn>): URL[] {
  return fetchMock.mock.calls
    .map(([input]) => new URL(String(input), 'http://localhost'))
    .filter((url) => url.pathname === '/api/ai-conversations');
}

describe('ActivityPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('keeps global conversation requests free of workspace_id even when override text is entered', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/overview') {
        return createJsonResponse({ recent_agent_runs: [] });
      }
      if (url.pathname === '/api/ai-conversations') {
        return createJsonResponse({ conversations: [] });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(<ActivityPage />);

    await waitFor(() => expect(getConversationRequests(fetchMock)).toHaveLength(1));

    fireEvent.change(screen.getByPlaceholderText('optional workspace-id override'), {
      target: { value: 'workspace-123' },
    });

    await waitFor(() => expect(getConversationRequests(fetchMock)).toHaveLength(1));

    const request = getConversationRequests(fetchMock).at(-1);
    expect(request?.searchParams.get('scope')).toBeNull();
    expect(request?.searchParams.get('workspace_id')).toBeNull();
  });

  it('sends workspace scope and workspace_id when workspace scope is selected', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/overview') {
        return createJsonResponse({ recent_agent_runs: [] });
      }
      if (url.pathname === '/api/ai-conversations') {
        return createJsonResponse({ conversations: [] });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(<ActivityPage />);

    await waitFor(() => expect(getConversationRequests(fetchMock)).toHaveLength(1));

    fireEvent.change(screen.getByDisplayValue('global conversations'), {
      target: { value: 'workspace' },
    });
    fireEvent.change(screen.getByPlaceholderText('optional workspace-id override'), {
      target: { value: ' workspace-789 ' },
    });

    await waitFor(() => {
      const request = getConversationRequests(fetchMock).at(-1);
      expect(request?.searchParams.get('scope')).toBe('workspace');
      expect(request?.searchParams.get('workspace_id')).toBe('workspace-789');
    });
  });
});