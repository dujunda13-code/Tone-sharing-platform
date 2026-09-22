import type { ReactNode } from "react";

/**
 * 页面主视觉：统一的页眉标题、说明与主操作区。
 */
export function PageHero({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-hero">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <h1>{title}</h1>
      {description && <p className="page-hero-description">{description}</p>}
      {actions && <div className="page-hero-actions">{actions}</div>}
    </header>
  );
}
