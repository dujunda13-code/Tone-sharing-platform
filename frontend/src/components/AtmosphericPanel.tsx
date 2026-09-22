import type { ReactNode } from "react";

/**
 * 氛围渐变面板：只负责柔和的径向渐变装饰，不承载状态语义。
 * tone 仅影响装饰配色，绝不用于按钮或文字。
 */
export function AtmosphericPanel({
  tone,
  children,
  className,
}: {
  tone: "mint" | "peach" | "lavender" | "sky" | "rose";
  children: ReactNode;
  className?: string;
}) {
  return <section className={`atmospheric-panel tone-${tone}${className ? ` ${className}` : ""}`}>{children}</section>;
}
