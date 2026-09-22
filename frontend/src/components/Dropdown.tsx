import { useEffect, useId, useRef, useState } from "react";

type Option = { value: string; label: string; disabled?: boolean };

/** 可见选项面板与键盘选择共享同一状态，替代浏览器不可定制的原生弹出层。 */
export function Dropdown({ label, value, options, onChange, disabled = false, className = "" }: {
  label: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const listId = useId();
  const enabled = options.filter((option) => !option.disabled);
  const chosen = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, [open]);

  const show = () => {
    setActive(Math.max(0, enabled.findIndex((option) => option.value === value)));
    setOpen(true);
  };

  const choose = (next: string) => {
    onChange(next);
    setOpen(false);
    root.current?.querySelector<HTMLButtonElement>(".dropdown-trigger")?.focus();
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === "Escape") { setOpen(false); return; }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!open) { show(); return; }
      setActive((index) => (index + (event.key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length);
    }
    if ((event.key === "Enter" || event.key === " ") && open) {
      event.preventDefault();
      if (enabled[active]) choose(enabled[active].value);
    }
  };

  return (
    <div className={`dropdown ${className}`} ref={root}>
      <button type="button" className="dropdown-trigger" role="combobox" aria-label={label} aria-haspopup="listbox" aria-expanded={open} aria-controls={listId} aria-activedescendant={open ? `${listId}-${active}` : undefined} disabled={disabled || enabled.length === 0} onClick={() => open ? setOpen(false) : show()} onKeyDown={onKeyDown}>
        <span>{chosen?.label ?? label}</span><svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 7 5 5 5-5" /></svg>
      </button>
      {open && (
        <div id={listId} className="dropdown-list" role="listbox" aria-label={label}>
          {options.map((option) => {
            const index = enabled.findIndex((item) => item.value === option.value);
            return <button key={option.value} id={index >= 0 ? `${listId}-${index}` : undefined} type="button" role="option" aria-selected={option.value === value} className={index === active ? "dropdown-option active" : "dropdown-option"} disabled={option.disabled} onMouseEnter={() => { if (index >= 0) setActive(index); }} onClick={() => choose(option.value)}>{option.label}{option.value === value && <span aria-hidden="true">✓</span>}</button>;
          })}
        </div>
      )}
    </div>
  );
}
