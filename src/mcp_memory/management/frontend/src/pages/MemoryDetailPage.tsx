import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { fetchMemoryDetail } from '../lib/api';

function renderMemoryReference(memoryId: string, label?: string | null) {
  const resolvedLabel = (label ?? memoryId) || '-';
  if (!memoryId || memoryId.startsWith('ext:')) {
    return <span>{resolvedLabel}</span>;
  }

  return (
    <Link className="text-accent hover:underline" to={`/memory/${memoryId}`} title={memoryId}>
      {resolvedLabel}
    </Link>
  );
}

export function MemoryDetailPage() {
  const { memoryId = '' } = useParams();
  const detailQuery = useQuery({
    queryKey: ['memory-detail', memoryId],
    queryFn: () => fetchMemoryDetail(memoryId),
    enabled: Boolean(memoryId),
  });

  if (detailQuery.isPending) {
    return <section className="panel p-6 text-muted">Loading memory detail…</section>;
  }
  if (detailQuery.isError || !detailQuery.data) {
    return <section className="panel p-6 text-danger">Unable to load memory detail.</section>;
  }

  const { record, relationships, superseded } = detailQuery.data;

  return (
    <div className="space-y-6">
      <section className="panel p-4">
        <p className="panel-title">Memory detail</p>
        <h2 className="mt-1 text-lg font-semibold text-text">{record.title}</h2>
        <div className="mt-2 flex flex-wrap gap-3 text-sm text-muted">
          <span>{record.type}</span>
          <span>{record.status}</span>
          <span>{record.updated_at}</span>
        </div>
        <pre className="mt-6 whitespace-pre-wrap break-words text-sm text-text">{record.content}</pre>
      </section>

      <section className="grid gap-6 xl:grid-cols-3">
        <div className="table-shell xl:col-span-2">
          <div className="border-b border-border px-4 py-4">
            <p className="panel-title">Relationships</p>
            <h2 className="mt-2 text-lg font-semibold text-text">Incoming and outgoing links</h2>
          </div>
          <table>
            <thead>
              <tr>
                <th>Direction</th>
                <th>Link type</th>
                <th>Source</th>
                <th>Target</th>
                <th>Context</th>
              </tr>
            </thead>
            <tbody>
              {[...relationships.incoming.map((link) => ({ direction: 'incoming', ...link })), ...relationships.outgoing.map((link) => ({ direction: 'outgoing', ...link }))].map((link, index) => (
                <tr key={`${link.direction}-${link.source_id}-${link.target_id}-${index}`}>
                  <td>{link.direction}</td>
                  <td>{link.link_type}</td>
                  <td>{renderMemoryReference(link.source_id)}</td>
                  <td>{renderMemoryReference(link.target_id)}</td>
                  <td>{link.context || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-shell">
          <div className="border-b border-border px-4 py-4">
            <p className="panel-title">Superseded</p>
            <h2 className="mt-2 text-lg font-semibold text-text">Lineage breadcrumbs</h2>
          </div>
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {superseded.length ? superseded.map((item) => (
                <tr key={item.id}>
                  <td>{renderMemoryReference(item.id, item.title)}</td>
                  <td>{item.type}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={2} className="text-muted">No superseded records.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}