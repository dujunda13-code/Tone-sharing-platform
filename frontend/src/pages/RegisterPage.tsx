import { useState } from "react";

import { api, getApiError } from "../api/client";
import { AtmosphericPanel } from "../components/AtmosphericPanel";
import { AtmosphericWave } from "../components/AtmosphericWave";
import { BrandGlyph } from "../components/BrandGlyph";
import { ErrorPanel } from "../components/ErrorPanel";
import { PasswordField } from "../components/PasswordField";

export function RegisterPage({ onRegistered, onBack }: { onRegistered: () => void; onBack: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState<{ code: string; message: string; requestId: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (password !== confirmation) {
      setError({ code: "PASSWORD_MISMATCH", message: "两次密码不一致", requestId: "" });
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await api.register(username, password);
      onRegistered();
    } catch (requestError) {
      setError(getApiError(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth-layout auth-preview auth-preview-register">
      <AtmosphericPanel tone="lavender" className="auth-art-panel">
        <AtmosphericWave className="auth-preview-waves" />
        <span className="auth-brand-mark">
          <BrandGlyph />
          <span>音色共享平台</span>
        </span>
        <h1>创建账号</h1>
        <p>开启你的声音创作之旅。分享、连接，让好声音服务更多人。</p>
        <span className="auth-art-signature" aria-hidden="true">
          用声音<br />创造更多可能
        </span>
      </AtmosphericPanel>
      <section className="auth-form-panel auth-card" aria-label="注册">
        <h2 id="register-title">注册新账号</h2>
        <p className="page-hero-description">创建账号，开启专属声音创作。</p>
        <form className="auth-form" onSubmit={(event) => void submit(event)}>
          <div className="field">
            <label htmlFor="register-username">用户名</label>
            <input
              id="register-username"
              type="text"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              minLength={3}
              maxLength={64}
              required
            />
          </div>
          <PasswordField
            id="register-password"
            label="密码"
            value={password}
            onChange={setPassword}
            autoComplete="new-password"
          />
          <p className="hint">至少 8 位；密码不会保存在浏览器存储。</p>
          <PasswordField id="register-confirmation" label="确认密码" value={confirmation} onChange={setConfirmation} autoComplete="new-password" />
          <p className="auth-terms">注册即表示你已阅读并同意平台使用说明。</p>
          {error && <ErrorPanel error={error} />}
          <button className="primary" type="submit" disabled={submitting || !username || !password || !confirmation}>
            {submitting ? "注册中…" : "注册"}
          </button>
        </form>
        <div className="auth-actions">
          <button className="secondary" type="button" onClick={onBack}>返回登录</button>
        </div>
      </section>
    </main>
  );
}
