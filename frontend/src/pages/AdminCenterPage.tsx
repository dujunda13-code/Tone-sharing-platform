import { useCallback, useEffect, useState } from "react";

import { api, getApiError } from "../api/client";
import type { AdminUserSummary, AuditEventSummary, AuthUser } from "../api/types";
import { Disclosure } from "../components/Disclosure";
import type { SafeError } from "../components/ErrorPanel";
import { ErrorPanel } from "../components/ErrorPanel";
import { EmptyState } from "../components/EmptyState";
import { StatusBadge } from "../components/StatusBadge";

function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN", { hour12: false });
}

const ROLE_LABELS: Record<AdminUserSummary["role"], string> = {
  admin: "管理员",
  user: "普通用户",
};

const EVENT_TYPE_LABELS: Record<string, string> = {
  "admin.user_status_changed": "账号状态变更",
};

function eventLabel(eventType: string): string {
  return EVENT_TYPE_LABELS[eventType] ?? "管理操作";
}

const TABLE_COLUMNS = ["用户", "角色", "状态", "参考音频", "音色", "任务", "操作"] as const;

/**
 * 管理中心：仅 admin 角色挂载；前端门禁只是体验层，后端 require_admin 独立强制。
 * 桌面为语义表格（窄屏由响应式样式堆叠同一 DOM），审计事件按时间线性呈现；
 * 对象 ID 与事件元数据折叠进"事件详情"。
 */
export function AdminCenterPage({ user }: { user: AuthUser }) {
  const [users, setUsers] = useState<AdminUserSummary[] | null>(null);
  const [events, setEvents] = useState<AuditEventSummary[]>([]);
  const [nextBeforeId, setNextBeforeId] = useState<number | null>(null);
  const [error, setError] = useState<SafeError | null>(null);
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    if (user.role !== "admin") return;
    let active = true;
    void api
      .adminUsers()
      .then((list) => {
        if (!active) return;
        setUsers(list.items);
        setError(null);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setUsers(null);
        setError(getApiError(requestError));
      });
    return () => {
      active = false;
    };
  }, [user.role, reloadToken]);

  useEffect(() => {
    if (user.role !== "admin") return;
    let active = true;
    void api
      .adminAuditEvents()
      .then((list) => {
        if (!active) return;
        setEvents(list.items);
        setNextBeforeId(list.next_before_id);
      })
      .catch(() => {
        /* 审计列表失败不阻塞用户管理；保持空态 */
      });
    return () => {
      active = false;
    };
  }, [user.role]);

  const changeStatus = async (target: AdminUserSummary) => {
    setBusyUserId(target.id);
    setError(null);
    try {
      await api.updateAdminUserStatus(target.id, target.status === "active" ? "disabled" : "active");
      const list = await api.adminUsers();
      setUsers(list.items);
    } catch (requestError) {
      setError(getApiError(requestError));
    } finally {
      setBusyUserId(null);
    }
  };

  const loadEarlier = async () => {
    if (nextBeforeId === null) return;
    const list = await api.adminAuditEvents(20, nextBeforeId).catch(() => null);
    if (!list) return;
    // 游标翻页可能返回重叠窗口，按事件 id 去重，避免重复记录与重复 key。
    setEvents((previous) => {
      const known = new Set(previous.map((event) => event.id));
      return [...previous, ...list.items.filter((event) => !known.has(event.id))];
    });
    setNextBeforeId(list.next_before_id);
  };

  if (user.role !== "admin") {
    return (
      <section className="card">
        <h2>管理中心</h2>
        <EmptyState title="仅管理员可访问" description="该页面仅对管理员开放。" />
      </section>
    );
  }

  return (
    <div className="page-stack preview-page admin-center-page">
      <section className="card page-intro">
        <h2>管理中心</h2>
        <p className="card-lede">
          查看各账号的资源归属与审计记录；可停用或恢复其他账号，操作会写入审计。
        </p>
      </section>

      {error && (
        <ErrorPanel error={error} hint="管理接口由后端保护；未就绪时保持失败提示。" />
      )}

      <div className="admin-board">
      <section className="card admin-user-panel">
        <header className="resource-card-head">
          <h3>用户列表</h3>
          <button className="secondary" type="button" onClick={retry}>
            刷新
          </button>
        </header>
        {users === null && !error && <p aria-live="polite">正在加载用户…</p>}
        {users !== null && users.length === 0 && <EmptyState title="暂无用户记录" />}
        {users !== null && users.length > 0 && (
          <table aria-label="用户列表">
            <thead>
              <tr>
                {TABLE_COLUMNS.map((column) => (
                  <th key={column} scope="col">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {users.map((item) => (
                <tr key={item.id}>
                  <td data-label="用户">{item.username}</td>
                  <td data-label="角色">{ROLE_LABELS[item.role] ?? item.role}</td>
                  <td data-label="状态">
                    <StatusBadge status={item.status} />
                  </td>
                  <td data-label="参考音频">{item.dataset_count}</td>
                  <td data-label="音色">{item.voice_count}</td>
                  <td data-label="任务">{item.job_count}</td>
                  <td data-label="操作">
                    {item.id === user.id ? (
                      <button className="secondary" type="button" disabled>
                        当前账号
                      </button>
                    ) : (
                      <button
                        className="secondary"
                        type="button"
                        disabled={busyUserId === item.id}
                        onClick={() => void changeStatus(item)}
                      >
                        {busyUserId === item.id
                          ? "处理中…"
                          : item.status === "active"
                            ? "停用"
                            : "恢复"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card admin-audit-panel">
        <header className="resource-card-head">
          <h3>审计记录</h3>
          {nextBeforeId !== null && (
            <button className="secondary" type="button" onClick={() => void loadEarlier()}>
              加载更早事件
            </button>
          )}
        </header>
        {events.length === 0 ? (
          <EmptyState title="暂无审计记录" description="管理员操作和关键安全事件会按时间记录在这里。" />
        ) : (
          <ol className="audit-timeline">
            {events.map((event) => (
              <li key={event.id}>
                <span className="job-kind">{eventLabel(event.event_type)}</span>
                <span className="job-time">{formatTime(event.created_at)}</span>
                <Disclosure label="事件详情">
                  {event.event_type}
                  {event.subject_id ? ` · 对象 ID：${event.subject_id}` : ""}
                  {Object.keys(event.metadata).length > 0
                    ? ` · ${Object.entries(event.metadata)
                        .map(([key, value]) => `${key}=${value}`)
                        .join(" · ")}`
                    : ""}
                </Disclosure>
              </li>
            ))}
          </ol>
        )}
      </section>
      </div>
    </div>
  );
}
