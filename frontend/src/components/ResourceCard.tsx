import type { ReactNode } from "react";

import { StatusBadge } from "./StatusBadge";

export function ResourceCard({
  title,
  status,
  meta,
  action,
  children,
}: {
  title: ReactNode;
  status?: string;
  meta?: Array<{ label: string; value: ReactNode }>;
  action?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <section className="card resource-card">
      <header className="resource-card-head">
        <h3>{title}</h3>
        {status && <StatusBadge status={status} />}
      </header>
      {meta && meta.length > 0 && (
        <dl className="kv">
          {meta.map((item) => (
            <FragmentRow key={item.label} label={item.label} value={item.value} />
          ))}
        </dl>
      )}
      {children}
      {action && <div className="resource-card-actions">{action}</div>}
    </section>
  );
}

function FragmentRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}
