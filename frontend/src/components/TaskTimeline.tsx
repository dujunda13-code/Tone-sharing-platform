import type { JobKind, JobSummary } from "../api/types";
import { toUserFacingMessage } from "../api/client";
import { Disclosure } from "./Disclosure";
import { StatusBadge } from "./StatusBadge";

const KIND_LABELS: Record<JobKind, string> = {
  preprocess: "音频分析",
  train: "历史训练",
  synthesize: "语音合成",
  evaluate: "质量评测",
};

export function formatDuration(startedAt: string | null, finishedAt: string | null): string {
  if (!startedAt) return "尚未开始";
  const end = finishedAt ? new Date(finishedAt).getTime() : Date.now();
  const seconds = Math.max(0, Math.round((end - new Date(startedAt).getTime()) / 1000));
  return `${seconds} 秒`;
}

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN", { hour12: false });
}

function durationText(job: JobSummary): string {
  if (job.started_at === null) return "尚未开始";
  return job.finished_at === null ? "进行中" : `耗时 ${formatDuration(job.started_at, job.finished_at)}`;
}

/** 任务时间线：自然语言结论 + 固定状态标签；任务 ID 与错误代码折叠进"查看详情"。 */
export function TaskTimeline({ jobs }: { jobs: JobSummary[] }) {
  return (
    <ul className="task-timeline">
      {jobs.map((job) => (
        <li key={job.id}>
          <div className="task-head">
            <StatusBadge status={job.status} />
            <span className="job-kind">{KIND_LABELS[job.kind] ?? job.kind}</span>
            <span className="task-time">{durationText(job)}</span>
          </div>
          <p className="task-message">
            {job.public_message
              ? toUserFacingMessage(job.public_message)
              : job.error_code
                ? "任务未完成，详情可展开查看原因。"
                : "正在等待系统处理。"}
          </p>
          <p className="task-times">
            创建 {formatTime(job.created_at)} · 开始 {formatTime(job.started_at)} · 结束 {formatTime(job.finished_at)}
          </p>
          <Disclosure label="查看详情">
            任务 ID：{job.id}
            {job.error_code ? ` · 错误代码：${job.error_code}` : ""}
          </Disclosure>
        </li>
      ))}
    </ul>
  );
}
