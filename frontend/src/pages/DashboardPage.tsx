import { useCallback, useEffect, useState } from "react";

import { api, getApiError, toUserFacingMessage } from "../api/client";
import type { DashboardResponse, JobKind, VoiceSummary } from "../api/types";
import type { SafeError } from "../components/ErrorPanel";
import { ErrorPanel } from "../components/ErrorPanel";
import { EmptyState } from "../components/EmptyState";
import { AtmosphericPanel } from "../components/AtmosphericPanel";
import { AtmosphericWave } from "../components/AtmosphericWave";
import { StatusBadge } from "../components/StatusBadge";
import type { WorkspacePage } from "../components/AppShell";

const KIND_LABELS: Record<JobKind, string> = {
  preprocess: "音频分析",
  train: "历史训练",
  synthesize: "语音合成",
  evaluate: "质量评测",
};

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN", { hour12: false });
}

function voiceInitials(voice: VoiceSummary): string {
  const name = voice.display_name.trim();
  return name ? name.slice(0, 1) : "声";
}

function VoiceRow({ voice, onPlay }: { voice: VoiceSummary; onPlay: () => void }) {
  return (
    <li className="voice-row">
      <span className="voice-icon" aria-hidden="true">
        {voiceInitials(voice)}
      </span>
      <div className="voice-row-main">
        <div className="voice-row-title">{voice.display_name}</div>
        <div className="voice-row-meta">
          {voice.mode === "zero_shot" ? "零样本" : "历史档案"} · {formatTime(voice.created_at)} ·{" "}
          {voice.can_synthesize ? "可用于语音创作" : "准备中"}
        </div>
      </div>
      <button
        className="voice-play-button"
        type="button"
        aria-label={`试听 ${voice.display_name}`}
        disabled={!voice.can_synthesize}
        onClick={onPlay}
      >
        ▶
      </button>
      <StatusBadge status={voice.status} />
    </li>
  );
}

/** 工作台：编辑式看板。概览计数、最近任务与系统状态全部来自真实接口；四态齐备。 */
export function DashboardPage({ onNavigate }: { onNavigate: (page: WorkspacePage) => void }) {
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [error, setError] = useState<SafeError | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloadToken, setReloadToken] = useState(0);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    void api.dashboard()
      .then((overview) => {
        if (!active) return;
        setDashboard(overview);
        setError(null);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setDashboard(null);
        setError(getApiError(requestError));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [reloadToken]);

  if (loading) {
    return (
      <section className="card" aria-live="polite">
        <h2>正在加载工作台…</h2>
      </section>
    );
  }

  if (error) {
    return (
      <section className="card">
        <h2>工作台加载失败</h2>
        <ErrorPanel error={error} hint="概览接口由后端提供；未就绪时保持失败提示，不会显示虚构数据。" />
        <button className="secondary" type="button" onClick={retry}>
          重试
        </button>
      </section>
    );
  }

  if (!dashboard) return null;

  const readiness = dashboard.readiness;
  const systemOk = readiness.status === "ready";
  const pendingChecks = Object.values(readiness.checks).filter((check) => !check.ok).length;

  return (
    <div className="dashboard preview-dashboard">
      <div className="dashboard-hero">
        <AtmosphericPanel tone="mint">
        <AtmosphericWave className="dashboard-hero-waves" />
        <p className="eyebrow">工作台</p>
        <h1>让每一种声音，都能被清晰表达。</h1>
        <p>用一段授权的 3–10 秒参考音频创建音色，随时开始语音创作。</p>
        <div className="dash-actions">
          <button className="primary" type="button" onClick={() => onNavigate("create")}>
            创建音色
          </button>
          <button className="secondary" type="button" onClick={() => onNavigate("synthesize")}>
            开始合成
          </button>
        </div>
        <span className="hero-signature" aria-hidden="true">声音，让世界更近</span>
        </AtmosphericPanel>
      </div>

      <section className="stat-grid" aria-label="资源概览">
        <div className="stat-card metric-card">
          <span className="metric-icon metric-icon-mint" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M9 3h6l1 3h3v5l2 1-2 2v5h-3l-1 2H9l-1-2H5v-5l-2-2 2-1V6h3l1-3Z"/><path d="M9 9h6v6H9z"/></svg></span>
          <span className="stat-value">{dashboard.counts.datasets}</span>
          <span className="stat-label">参考音频</span>
        </div>
        <div className="stat-card metric-card">
          <span className="metric-icon metric-icon-sky" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="12" cy="7" r="3"/><path d="M5.5 20v-2a6.5 6.5 0 0 1 13 0v2H5.5Z"/></svg></span>
          <span className="stat-value">{dashboard.counts.voices}</span>
          <span className="stat-label">我的音色</span>
        </div>
        <div className="stat-card metric-card">
          <span className="metric-icon metric-icon-peach" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 7v5l3.5 2"/></svg></span>
          <span className="stat-value">{dashboard.counts.active_jobs}</span>
          <span className="stat-label">处理中任务</span>
        </div>
        <div className="stat-card metric-card">
          <span className="metric-icon metric-icon-rose" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M5 20v-7m5 7V5m5 15V9m5 11V3"/></svg></span>
          <span className="stat-value">{dashboard.counts.syntheses}</span>
          <span className="stat-label">合成作品</span>
        </div>
      </section>

      <div className="dashboard-lower-grid dashboard-board-grid">
        <div className="dashboard-primary-column">
      <section className="card dashboard-system-card">
        <header className="resource-card-head">
          <h2>系统状态</h2>
          <span className={`ready-light${systemOk ? " ok" : ""}`}>
            <span className="dot" aria-hidden="true" />
            {systemOk ? "系统可用" : "需要检查"}
          </span>
        </header>
        {systemOk ? (
          <p className="note">系统就绪，可以创建音色并进行零样本合成。</p>
        ) : (
          <p className="note">
            还有 {pendingChecks} 项系统检查未通过（如基础模型或转写模型缺失）。未就绪前合成请求会按安全策略拒绝，界面不会显示虚假的可用状态。
          </p>
        )}
        <button className="secondary" type="button" onClick={() => onNavigate("tasks")}>
          查看任务中心
        </button>
      </section>

      <section className="card dashboard-recent-card">
        <header className="resource-card-head">
          <h2>最近任务</h2>
          <button className="secondary" type="button" onClick={() => onNavigate("tasks")}>
            全部任务
          </button>
        </header>
        {dashboard.recent_jobs.length === 0 ? (
          <EmptyState
            title="还没有任务"
            description="创建音色或开始合成后，这里会显示真实的任务状态。"
            action={
              <button className="secondary" type="button" onClick={() => onNavigate("create")}>
                去创建音色
              </button>
            }
          />
        ) : (
          <ul className="job-list">
            {dashboard.recent_jobs.map((job) => (
              <li key={job.id}>
                <StatusBadge status={job.status} />
                <span className="job-kind">{KIND_LABELS[job.kind] ?? job.kind}</span>
                <span className="job-message">
                  {job.public_message
                    ? toUserFacingMessage(job.public_message)
                    : job.error_code
                      ? "任务未完成，详情可在任务中心查看。"
                      : "—"}
                </span>
                <span className="job-time">{formatTime(job.created_at)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
        </div>

      <section className="card dashboard-voices-card" role="region" aria-label="我的音色">
        <header className="resource-card-head">
          <h2>我的音色</h2>
          <button className="secondary" type="button" onClick={() => onNavigate("voices")}>
            进入音色库
          </button>
        </header>
        {dashboard.voices.length === 0 ? (
          <EmptyState title="还没有音色档案" description="上传授权的 3–10 秒参考音频并完成审核后，可用音色会出现在这里。" />
        ) : (
          <ul className="voice-row-list">
            {dashboard.voices.map((voice) => (
              <VoiceRow key={voice.id} voice={voice} onPlay={() => onNavigate("synthesize")} />
            ))}
          </ul>
        )}
      </section>
      </div>
    </div>
  );
}
