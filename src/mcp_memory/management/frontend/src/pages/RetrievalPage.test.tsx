import { screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { RetrievalPage } from './RetrievalPage';
import { renderWithQueryClient } from '../test/utils';

describe('RetrievalPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('keeps the header chrome visible when retrieval metrics fail', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/metrics/nerd') {
        return {
          ok: false,
          status: 500,
          json: async () => ({ detail: 'boom' }),
        } as Response;
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(<RetrievalPage />);

    await waitFor(() => expect(screen.getByText('Unable to load retrieval metrics.')).toBeInTheDocument());

    expect(screen.getByText('Retrieval')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '24h' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '7d' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '30d' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Global' })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Workspace scope' })).toBeInTheDocument();
  });
});