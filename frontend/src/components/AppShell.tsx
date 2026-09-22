import { useEffect, useState, type ReactNode } from "react";

import type { AuthUser, HealthResponse } from "../api/types";
import { BrandGlyph } from "./BrandGlyph";
import { NotificationBell } from "./NotificationBell";
import { ReadinessPanel } from "./ReadinessPanel";

export type WorkspacePage =
  | "dashboard"
  | "plaza"
  | "voices"
  | "create"
  | "synthesize"
  | "tasks"
  | "safety"
  | "admin"
  | "profile";

export const WORKSPACE_NAV: Array<{ id: WorkspacePage; label: string; adminOnly?: boolean }> = [
  { id: "dashboard", label: "工作台" },
  { id: "plaza", label: "音色广场" },
  { id: "voices", label: "我的音色" },
  { id: "create", label: "创建音色" },
  { id: "synthesize", label: "语音创作" },
  { id: "tasks", label: "任务中心" },
  { id: "safety", label: "安全中心" },
  { id: "admin", label: "管理中心", adminOnly: true },
];

function SearchGlyph() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="10.8" cy="10.8" r="6.4" />
      <path d="m16 16 4.4 4.4" />
    </svg>
  );
}

export function readinessSummary(health: HealthResponse | null): {
  ok: boolean;
  label: string;
} {
  if (!health) return { ok: false, label: "检查中" };
  return health.status === "ready"
    ? { ok: true, label: "系统可用" }
    : { ok: false, label: "需要检查" };
}

/**
 * 应用壳：64px 顶部导航（角色过滤）、系统状态浮层（固定在导航下方，绝不被内容
 * 遮挡）、移动端折叠菜单。页面内容、数据加载与错误处理由 children 负责。
 */
export function AppShell({
  user,
  health,
  activePage,
  onNavigate,
  onLogout,
  onOpenNotifications = () => {},
  children,
}: {
  user: AuthUser;
  health: HealthResponse | null;
  activePage: WorkspacePage;
  onNavigate: (page: WorkspacePage) => void;
  onLogout: () => Promise<void>;
  onOpenNotifications?: () => void;
  children: ReactNode;
}) {
  const [showChecks, setShowChecks] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const items = WORKSPACE_NAV.filter((item) => !item.adminOnly || user.role === "admin");
  const readiness = readinessSummary(health);

  useEffect(() => {
    if (!showChecks) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setShowChecks(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [showChecks]);

  const navigate = (page: WorkspacePage) => {
    setShowChecks(false);
    setMenuOpen(false);
    onNavigate(page);
  };

  const navButtons = (onPick: (page: WorkspacePage) => void) =>
    items.map((item) => (
      <button
        key={item.id}
        type="button"
        className={`top-nav-link${activePage === item.id ? " active" : ""}`}
        aria-current={activePage === item.id ? "page" : undefined}
        onClick={() => onPick(item.id)}
      >
        {item.label}
      </button>
    ));

  return (
    <div className="app-shell preview-app-shell preview-app-shell-fluid">
      <header className="top-nav preview-top-nav">
        <div className="top-nav-inner">
          <span className="brand-mark" aria-label="音色共享平台">
            <BrandGlyph />
            <span>音色共享平台</span>
          </span>
          <nav className="top-nav-links preview-nav" aria-label="主导航">
            {navButtons(navigate)}
          </nav>
          <div className="top-nav-side readiness-anchor">
            <button className="nav-icon-button" type="button" aria-label="搜索">
              <SearchGlyph />
            </button>
            <NotificationBell onViewAll={onOpenNotifications} />
            <button
              type="button"
              className={`ready-light${readiness.ok ? " ok" : ""}`}
              aria-label="查看系统状态"
              aria-expanded={showChecks}
              aria-haspopup="dialog"
              onClick={() => setShowChecks((value) => !value)}
            >
              <span className="dot" aria-hidden="true" />
              {readiness.label}
            </button>
            {showChecks && <ReadinessPanel readiness={health} />}
            <button type="button" className="user-chip" aria-label="打开个人中心" onClick={() => navigate("profile")}>
              <span className="user-avatar" aria-hidden="true">{user.username.slice(0, 1).toUpperCase()}</span>
              <span className="user-name">{user.username}</span>
              <span className="role-badge">{user.role === "admin" ? "管理员" : "成员"}</span>
            </button>
            <button className="secondary nav-logout" type="button" onClick={() => void onLogout()}>
              退出登录
            </button>
            <button
              type="button"
              className="menu-toggle"
              aria-controls="mobile-nav"
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen((value) => !value)}
            >
              打开导航菜单
            </button>
          </div>
        </div>
      </header>
      {menuOpen && (
        <nav className="mobile-nav" id="mobile-nav" aria-label="移动端导航">
          {navButtons(navigate)}
        </nav>
      )}
      <main className="workspace-content">{children}</main>
    </div>
  );
}
