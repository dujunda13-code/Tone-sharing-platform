import { useEffect, useState } from "react";

import { api, getApiError, toUserFacingMessage } from "../api/client";
import type { JobSummary } from "../api/types";
import { Disclosure } from "./Disclosure";
import type { SafeError } from "./ErrorPanel";
import { ErrorPanel } from "./ErrorPanel";
import { StatusBadge } from "./StatusBadge";

const TERMINAL = new Set(["succeeded", "failed", "canceled"]);

/**
 * 任务进度：轮询安全摘要列表并按 jobId 匹配，只显示固定状态文案与
 * 自然语言进度；任务 ID、错误代码等内部信息折叠进"技术详情"。
 */
export function JobProgress({ jobId, label = "任务" }: { jobId: string; label?: string }) {
  const [job, setJob] = useState<JobSummary | null>(null);
  const [error, setError] = useState<SafeError | null>(null);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const list = await api.jobs();
        if (!active) return;
        const next = list.items.find((item) => item.id === jobId) ?? null;
        setJob(next);
        setError(null);
        return next?.status ?? null;
      } catch (requestError) {
        if (active) setError(getApiError(requestError));
        return "failed";
      }
    };

    void poll().then((status) => {
      if (!active || TERMINAL.has(status ?? "")) return;
      timer = window.setInterval(() => {
        void poll().then((nextStatus) => {
          if (TERMINAL.has(nextStatus ?? "") && timer !== undefined) window.clearInterval(timer);
        });
      }, 1000);
    });

    return () => {
      active = false;
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [jobId]);

  return (
    <section className="job-progress" aria-live="polite">
      <header>
        <strong>{label}</strong>
        <StatusBadge status={job?.status ?? "queued"} />
      </header>
      <Disclosure label="技术详情">
        任务 ID：{jobId}
        {job?.error_code ? ` · 错误代码：${job.error_code}` : ""}
      </Disclosure>
      {job?.public_message && <p>{toUserFacingMessage(job.public_message)}</p>}
      {!job && !error && <span>正在获取任务状态…</span>}
      {error && <ErrorPanel error={error} />}
    </section>
  );
}
