import { useId, useState } from "react";

function PasswordVisibilityGlyph({ visible }: { visible: boolean }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      {visible ? (
        <>
          <path d="M3.5 12s3.2-5 8.5-5 8.5 5 8.5 5-3.2 5-8.5 5-8.5-5-8.5-5Z" />
          <circle cx="12" cy="12" r="2.2" />
        </>
      ) : (
        <>
          <path d="m4 4 16 16" />
          <path d="M9.4 6.1A10.8 10.8 0 0 1 12 5.8c5.3 0 8.5 5 8.5 5a15.4 15.4 0 0 1-2.8 3.1M6.4 8.5A15.8 15.8 0 0 0 3.5 12s3.2 5 8.5 5c.8 0 1.6-.1 2.3-.4" />
        </>
      )}
    </svg>
  );
}

/**
 * 稳定的密码可见性控件：切换不依赖输入焦点，失焦后仍可再次点击。
 */
export function PasswordField({
  id,
  label,
  value,
  onChange,
  autoComplete,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  autoComplete?: string;
}) {
  const [visible, setVisible] = useState(false);
  const hintId = useId();
  return (
    <div className="field password-field-wrap">
      <label htmlFor={id}>{label}</label>
      <div className="password-field">
        <input
          id={id}
          type={visible ? "text" : "password"}
          value={value}
          autoComplete={autoComplete}
          aria-describedby={hintId}
          onChange={(event) => onChange(event.target.value)}
        />
        <button
          type="button"
          className="password-toggle"
          aria-label={visible ? `隐藏${label}` : `显示${label}`}
          aria-pressed={visible}
          onClick={() => setVisible((current) => !current)}
        >
          <PasswordVisibilityGlyph visible={visible} />
        </button>
      </div>
      <span id={hintId} className="sr-only-label">
        密码显示状态可随时切换
      </span>
    </div>
  );
}
