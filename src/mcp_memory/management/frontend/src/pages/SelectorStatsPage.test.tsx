import { screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SelectorStatsPage } from './SelectorStatsPage';
import { renderWithQueryClient } from '../test/utils';

function createJsonResponse(payload: unknown): Response {
  return {
    ok: true,
    json: async () => payload,
  } as Response;
}

describe('SelectorStatsPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders the quiet-window state without crashing when selector stats resolve after an initial pending render', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), 'http://localhost');
      if (url.pathname === '/api/metrics/nerd') {
        return createJsonResponse({
          composition: {
            by_workspace: [],
          },
        });
      }
      if (url.pathname === '/api/selector-stats') {
        return createJsonResponse({
          generated_at: 1,
          window_hours: 24,
          run_limit: 200,
          summary: {
            total_runs: 0,
            selector_signal_runs: 0,
            fresh_selector_runs: 0,
            seeded_claimed_runs: 0,
            unknown_runs: 0,
            fallback_runs: 0,
            mutation_runs: 0,
            no_op_runs: 0,
            total_mutations: 0,
            total_tool_calls: 0,
            average_candidate_count: null,
          },
          classification_breakdown: [],
          outcome_rows: [],
          feature_rollup_rows: [],
          recent_runs: [],
        });
      }
      throw new Error(`Unexpected request: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderWithQueryClient(<SelectorStatsPage />);

    await waitFor(() => expect(screen.getByText('Selector Stats')).toBeInTheDocument());
    expect(screen.getByText('Quiet window')).toBeInTheDocument();
    expect(screen.getByText(/No selector runs landed in this window yet/i)).toBeInTheDocument();
  });
});