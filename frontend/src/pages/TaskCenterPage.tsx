import { useCallback, useEffect, useState } from "react";

import { api, getApiError } from "../api/client";
import type { JobStatus, JobSummary } from "../api/types";
import { taskFilterDraftStore, useDraft } from "../state/drafts";
import type { SafeError } from "../components/ErrorPanel";
import { ErrorPanel } from "../components/ErrorPanel";
import { EmptyState } from "../components/EmptyState";
import { TaskTimeline } from "../components/TaskTimeline";

const STATUS_FILTERS: Array<{ label: string; value: JobStatus | undefined }> = [
  { label: "全部", value: undefined },
  { label: "排队中", value: "queued" },
  { label: "执行中", value: "running" },
  { label: "已完成", value: "succeeded" },
  { label: "已失败", value: "failed" },
];

/** 任务中心：当前用户全部预处理/训练/合成/评测任务的安全摘要与状态筛选。 */
export function TaskCenterPage() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [filterDraft, setFilterDraft] = useDraft(taskFilterDraftStore);
  const status = filterDraft.status;
  const [error, setError] = useState<SafeError | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    let active = true;
    void api
      .jobs(status)
      .then((list) => {
        if (!active) return;
        setJobs(list.items);
        setError(null);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setJobs(null);
        setError(getApiError(requestError));
      });
    return () => {
      active = false;
    };
  }, [status, reloadToken]);

  return (
    <div className="page-stack preview-page task-center-page">
      <section className="card page-intro task-board-heading">
        <h2>任务中心</h2>
        <p className="card-lede">
          这里汇总与你的账号相关的音频分析和语音合成任务；失败的任务会用通俗语言说明原因。
        </p>
        <div className="filter-row" role="group" aria-label="状态筛选">
          {STATUS_FILTERS.map((filter) => (
            <button
              key={filter.label}
              type="button"
              className={status === filter.value ? "secondary active-filter" : "secondary"}
              onClick={() => setFilterDraft({ status: filter.value })}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </section>
      <section className="card task-board" aria-label="任务列表">
        {jobs === null && error === null && (
          <p aria-live="polite">正在加载任务…</p>
        )}
        {error && (
          <>
            <ErrorPanel error={error} hint="任务列表接口由后端提供；未实现前保持失败提示。" />
            <button className="secondary" type="button" onClick={retry}>
              重试
            </button>
          </>
        )}
        {jobs !== null && !error && jobs.length === 0 && (
          <EmptyState title="还没有任务" description="创建音色或开始合成后，这里会显示任务状态。" />
        )}
        {jobs !== null && !error && jobs.length > 0 && <TaskTimeline jobs={jobs} />}
      </section>
    </div>
  );
}
