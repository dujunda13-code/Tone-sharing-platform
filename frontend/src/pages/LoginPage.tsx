import { useState } from "react";

import { api, getApiError } from "../api/client";
import type { AuthUser } from "../api/types";
import { AtmosphericPanel } from "../components/AtmosphericPanel";
import { BrandGlyph } from "../components/BrandGlyph";
import { ErrorPanel } from "../components/ErrorPanel";
import { PasswordField } from "../components/PasswordField";
import { RibbonBackground } from "../components/RibbonBackground";

export function LoginPage({
  onLogin,
  onRegister,
  notice,
}: {
  onLogin: (user: AuthUser) => void;
  onRegister: () => void;
  notice?: string;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<{ code: string; message: string; requestId: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      onLogin(await api.login(username, password));
    } catch (requestError) {
      setError(getApiError(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth-layout auth-preview auth-preview-login">
      <AtmosphericPanel tone="mint" className="auth-art-panel">
        <RibbonBackground className="auth-preview-ribbon" />
        <span className="auth-brand-mark">
          <BrandGlyph />
          <span>音色共享平台</span>
        </span>
        <h1>欢迎回来</h1>
        <p>声音因连接而清晰，让每一种声音都有被听见的可能。</p>
        <span className="auth-art-signature" aria-hidden="true">
          好声音<br />让世界更近
        </span>
      </AtmosphericPanel>
      <section className="auth-form-panel auth-card" aria-label="登录">
        <h2 id="login-title">登录账号</h2>
        <p className="page-hero-description">登录你的账号，继续创作属于你的声音。</p>
        {notice && <p className="notice-success">{notice}</p>}
        <form className="auth-form" onSubmit={(event) => void submit(event)}>
          <div className="field">
            <label htmlFor="login-username">用户名</label>
            <input
              id="login-username"
              type="text"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              required
            />
          </div>
          <PasswordField
            id="login-password"
            label="密码"
            value={password}
            onChange={setPassword}
            autoComplete="current-password"
          />
          <span className="auth-forgot-copy">忘记密码？</span>
          {error && <ErrorPanel error={error} />}
          <button className="primary" type="submit" disabled={submitting || !username || !password}>
            {submitting ? "登录中…" : "登录"}
          </button>
        </form>
        <div className="auth-actions auth-register-action">
          <span className="hint">还没有账号？</span>
          <button className="auth-register-link" type="button" onClick={onRegister}>立即注册</button>
        </div>
      </section>
    </main>
  );
}
