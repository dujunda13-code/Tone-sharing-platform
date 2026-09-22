import { useState } from "react";
import type { ReactNode } from "react";

/**
 * 技术详情统一折叠组件：内部 ID、错误码、路径等默认收起，用户按需展开。
 */
export function Disclosure({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="disclosure">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        {label}
      </button>
      {open && <div className="disclosure-body">{children}</div>}
    </div>
  );
}
