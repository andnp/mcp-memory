import { fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { LogsPage } from './LogsPage';
import { renderWithQueryClient } from '../test/utils';

function createJsonResponse(payload: unknown): Response {
  return {
    ok: true,
  } as Response;
}

function getRequests(fetchMock: ReturnType<typeof vi.fn>, path: string): URL[] {
  return fetchMock.mock.calls
    .map(([input]) => new URL(String(input), 'http://localhost'))
    .filter((url) => url.pathname === path);
}

describe('LogsPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('keeps global scope requests free of workspace_id even when override text is entered', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/logs/summary') {
        return createJsonResponse({ total: 0, by_level: {}, by_source: {} });
      }
      if (url.pathname === '/api/logs') {
        return createJsonResponse({ logs: [] });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(<LogsPage />);

    await waitFor(() => expect(getRequests(fetchMock, '/api/logs')).toHaveLength(1));

    fireEvent.change(screen.getByPlaceholderText('optional workspace-id override'), {
      target: { value: 'workspace-123' },
    });

    await waitFor(() => expect(getRequests(fetchMock, '/api/logs')).toHaveLength(1));

    const logRequest = getRequests(fetchMock, '/api/logs').at(-1);
    const summaryRequest = getRequests(fetchMock, '/api/logs/summary').at(-1);

    expect(logRequest?.searchParams.get('scope')).toBeNull();
    expect(logRequest?.searchParams.get('workspace_id')).toBeNull();
    expect(summaryRequest?.searchParams.get('scope')).toBeNull();
    expect(summaryRequest?.searchParams.get('workspace_id')).toBeNull();
  });

  it('sends workspace scope and workspace_id when workspace scope is selected', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/logs/summary') {
        return createJsonResponse({ total: 0, by_level: {}, by_source: {} });
      }
      if (url.pathname === '/api/logs') {
        return createJsonResponse({ logs: [] });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(<LogsPage />);

    await waitFor(() => expect(getRequests(fetchMock, '/api/logs')).toHaveLength(1));

    fireEvent.change(screen.getByDisplayValue('global logs'), {
      target: { value: 'workspace' },
    });
    fireEvent.change(screen.getByPlaceholderText('optional workspace-id override'), {
      target: { value: ' workspace-456 ' },
    });

    await waitFor(() => {
      const request = getRequests(fetchMock, '/api/logs').at(-1);
      expect(request?.searchParams.get('scope')).toBe('workspace');
      expect(request?.searchParams.get('workspace_id')).toBe('workspace-456');
    });

    const summaryRequest = getRequests(fetchMock, '/api/logs/summary').at(-1);
    expect(summaryRequest?.searchParams.get('scope')).toBe('workspace');
    expect(summaryRequest?.searchParams.get('workspace_id')).toBe('workspace-456');
  });
});