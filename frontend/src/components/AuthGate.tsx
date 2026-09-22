import { useEffect, useState, type ReactNode } from "react";

import { api } from "../api/client";
import type { AuthUser } from "../api/types";
import { LoginPage } from "../pages/LoginPage";
import { RegisterPage } from "../pages/RegisterPage";

export function AuthGate({
  children,
}: {
  children: (user: AuthUser, logout: () => Promise<void>) => ReactNode;
}) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [checking, setChecking] = useState(true);
  const [view, setView] = useState<"login" | "register">("login");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let active = true;
    void api.me()
      .then((current) => {
        if (active) setUser(current);
      })
      .catch(() => {
        if (active) setUser(null);
      })
      .finally(() => {
        if (active) setChecking(false);
      });
    return () => {
      active = false;
    };
  }, []);

  if (checking) {
    return (
      <main className="auth-layout">
        <section className="auth-form-panel" aria-live="polite">
          <p className="eyebrow">音色共享平台</p>
          <h1>正在进入…</h1>
        </section>
      </main>
    );
  }

  if (user) {
    return <>{children(user, async () => {
      await api.logout().catch(() => undefined);
      setUser(null);
      setView("login");
      setNotice("");
    })}</>;
  }

  if (view === "register") {
    return (
      <RegisterPage
        onBack={() => setView("login")}
        onRegistered={() => {
          setView("login");
          setNotice("注册成功，请使用新账号登录。");
        }}
      />
    );
  }

  return (
    <LoginPage
      notice={notice}
      onRegister={() => {
        setNotice("");
        setView("register");
      }}
      onLogin={(current) => {
        setNotice("");
        setUser(current);
      }}
    />
  );
}
