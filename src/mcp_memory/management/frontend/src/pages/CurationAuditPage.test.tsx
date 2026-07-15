import { fireEvent, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CurationAuditPage } from './CurationAuditPage';
import { renderWithQueryClient } from '../test/utils';

function createJsonResponse(payload: unknown): Response {
  return { ok: true, json: async () => payload } as Response;
}

const eventId = 'event-12345678';
const memoryId = 'memory-123';

describe('CurationAuditPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('loads bounded audit data and requires a second confirmation for restore', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/metrics/nerd') {
        return createJsonResponse({
          curation: {
            run_states: { terminal: 1 },
            history: { applied_event_count: 1, restorable_event_count: 1 },
            verified_yield: 1,
            specialist_routes: { total: 1, by_family: { taxonomist: 1 }, by_reason: { needs_different_specialist: 1 }, by_status: { open: 1 } },
            no_op_runs: 0,
          },
        });
      }
      if (url.pathname === '/api/mutation-history' && init?.method === 'POST') {
        return createJsonResponse({
          events: [{
            id: eventId,
            operation: 'normalize_memory',
            actor_kind: 'maintenance',
            family: 'curator',
            status: 'applied',
            created_at: '2026-07-15T12:00:00Z',
            receipt: { status: 'verified', operation: 'normalize_memory', action_id: 'action-1', run_id: 'run-1', affected_ids: [memoryId] },
          }],
          limit: 50,
          offset: 0,
          has_more: false,
          next_offset: null,
        });
      }
      if (url.pathname === `/api/mutation-history/${eventId}`) {
        return createJsonResponse({
          event: { id: eventId, operation: 'normalize_memory', actor_kind: 'maintenance', family: 'curator', status: 'applied' },
          receipt: { status: 'verified', operation: 'normalize_memory', action_id: 'action-1', run_id: 'run-1', affected_ids: [memoryId] },
          curation_run: { run_id: 'run-1', frontier_key: 'frontier', context_fingerprint: 'fingerprint', state: 'terminal', outcome: 'applied' },
          records: [],
          links: [],
          truncated: false,
        });
      }
      if (url.pathname === `/api/mutation-history/${eventId}/diff`) {
        return createJsonResponse({ event_id: eventId, records: [{ memory_id: memoryId, role: 'target', before: { title: 'Before' }, after: { title: 'After' }, before_exists: true, after_exists: true }], links: [], truncated: false });
      }
      if (url.pathname === `/api/mutation-history/${eventId}/restore-eligibility`) {
        return createJsonResponse({ event_id: eventId, eligible: true, operation: 'normalize_memory', inverse_operation: 'normalize_memory', risk: 'low', requires_confirmation: false, current_record_tokens: { [memoryId]: 'token-1' }, current_link_tokens: {}, protections: { [memoryId]: [] }, conflict_code: null, conflict_reason: null });
      }
      if (url.pathname === `/api/mutation-history/${eventId}/restore` && init?.method === 'POST') {
        return createJsonResponse({ status: 'applied', target_event_id: eventId, request_id: 'request-1', event_id: 'restore-1', conflict_reason: null, conflict_details: {} });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(<CurationAuditPage />);

    await waitFor(() => expect(screen.getByText('normalize_memory · event-12')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText('Before')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole('button', { name: 'Stage restore confirmation' })).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: 'Stage restore confirmation' }));
    expect(screen.getByText('Confirming creates a new restore event and never rewrites history.')).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([input, init]) => init?.method === 'POST' && String(input).includes('/restore')).length).toBe(0);

    fireEvent.click(screen.getByRole('button', { name: 'Confirm restore' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Restore request: applied.'));
  });
});
