import { screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import App from './App';
import { renderWithQueryClient } from './test/utils';

describe('App', () => {
  it('renders a not-found panel for unknown dashboard routes', async () => {
    renderWithQueryClient(
      <MemoryRouter basename="/dashboard" initialEntries={['/dashboard/totally-not-a-real-route']}>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByText("That dashboard page doesn't exist.")).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Go to overview' })).toHaveAttribute('href', '/dashboard');
  });
});