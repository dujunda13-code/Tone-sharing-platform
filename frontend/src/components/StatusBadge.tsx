const STATUS_LABELS: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  succeeded: "已完成",
  failed: "已失败",
  canceled: "已取消",
  created: "已创建",
  draft: "准备中",
  training: "训练中",
  ready: "可用",
  disabled: "已停用",
  active: "启用中",
  uploaded: "已上传",
  preprocessed: "质检完成",
};

/** 统一状态徽标：真实状态值的固定中文文案，未知状态原样显示，不推测。 */
export function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge status-${status}`}>{STATUS_LABELS[status] ?? status}</span>;
}
