import { fireEvent, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SearchPage } from './SearchPage';
import { renderWithQueryClient } from '../test/utils';

function createJsonResponse(payload: unknown): Response {
  return {
    ok: true,
    json: async () => payload,
  } as Response;
}

describe('SearchPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows a clear empty state after a successful search with no matches', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/memories/search') {
        return createJsonResponse({ results: [] });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(
      <MemoryRouter>
        <SearchPage />
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByPlaceholderText('Search by topic, person, memory title, or concept'), {
      target: { value: 'quiet graph' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));

    await waitFor(() => expect(screen.getByText('No memories matched “quiet graph”.')).toBeInTheDocument());
  });

  it('shows a clear error state when search fails', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/memories/search') {
        return {
          ok: false,
          status: 500,
          json: async () => ({ detail: 'boom' }),
        } as Response;
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(
      <MemoryRouter>
        <SearchPage />
      </MemoryRouter>,
    );

    fireEvent.change(screen.getByPlaceholderText('Search by topic, person, memory title, or concept'), {
      target: { value: 'broken search' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));

    await waitFor(() => expect(screen.getByText('Search failed. Try again in a moment.')).toBeInTheDocument());
  });
});