import { screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { MemoryDetailPage } from './MemoryDetailPage';
import { renderWithQueryClient } from '../test/utils';

function createJsonResponse(payload: unknown): Response {
  return {
    ok: true,
    json: async () => payload,
  } as Response;
}

describe('MemoryDetailPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders navigable links for related and superseded memory records', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/memories/memory-123') {
        return createJsonResponse({
          record: {
            id: 'memory-123',
            title: 'Primary memory',
            content: 'Details',
            summary: null,
            type: 'fact',
            status: 'active',
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-02T00:00:00Z',
            workspace_ids: ['workspace-a'],
            tags: [],
            metadata: {},
          },
          relationships: {
            incoming: [
              {
                source_id: 'memory-456',
                target_id: 'memory-123',
                link_type: 'REFERENCES',
                context: 'used for context',
              },
            ],
            outgoing: [
              {
                source_id: 'memory-123',
                target_id: 'ext:README.md',
                link_type: 'REFERENCES',
                context: 'documented here',
              },
            ],
          },
          superseded: [
            {
              id: 'memory-789',
              title: 'Legacy memory',
              summary: null,
              type: 'fact',
              status: 'stale',
            },
          ],
        });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(
      <MemoryRouter initialEntries={['/memory/memory-123']}>
        <Routes>
          <Route path="/memory/:memoryId" element={<MemoryDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getByText('Primary memory')).toBeInTheDocument());

    expect(screen.getByRole('link', { name: 'memory-456' })).toHaveAttribute('href', '/memory/memory-456');
    expect(screen.getAllByRole('link', { name: 'memory-123' })).toHaveLength(2);
    expect(screen.getAllByRole('link', { name: 'memory-123' })[0]).toHaveAttribute('href', '/memory/memory-123');
    expect(screen.getByRole('link', { name: 'Legacy memory' })).toHaveAttribute('href', '/memory/memory-789');
    expect(screen.getByText('ext:README.md')).toBeInTheDocument();
  });

  it('renders an explicit empty relationships row when no incoming or outgoing links exist', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/memories/memory-empty') {
        return createJsonResponse({
          record: {
            id: 'memory-empty',
            title: 'Lonely memory',
            content: 'Still valid',
            summary: null,
            type: 'fact',
            status: 'active',
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-02T00:00:00Z',
            workspace_ids: ['workspace-a'],
            tags: [],
            metadata: {},
          },
          relationships: {
            incoming: [],
            outgoing: [],
          },
          superseded: [],
        });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(
      <MemoryRouter initialEntries={['/memory/memory-empty']}>
        <Routes>
          <Route path="/memory/:memoryId" element={<MemoryDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => expect(screen.getByText('Lonely memory')).toBeInTheDocument());

    expect(screen.getByText('No incoming or outgoing links.')).toBeInTheDocument();
  });
});