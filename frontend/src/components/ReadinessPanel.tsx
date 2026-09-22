import type { HealthResponse } from "../api/types";

const CHECK_LABELS: Record<string, string> = {
  database: "SQLite 数据库",
  storage: "存储",
  gpu: "GPU",
  gpt_sovits: "GPT-SoVITS",
  emotion2vec: "Emotion2Vec",
  audioseal: "AudioSeal 水印",
};

/**
 * 系统状态浮层：逐项展示真实检查结论（可用 / 需要检查 / 等待响应）。
 * 状态文案由前端持有，不透传后端实现细节或路径。
 */
export function ReadinessPanel({ readiness }: { readiness: HealthResponse | null }) {
  return (
    <div
      className="readiness-panel readiness-popover"
      role="dialog"
      aria-label="系统检查"
      data-testid="readiness-panel"
    >
      <h3>系统检查</h3>
      <ul>
        {Object.entries(CHECK_LABELS).map(([key, label]) => {
          const check = readiness?.checks?.[key as keyof HealthResponse["checks"]];
          const state = !check ? "等待响应" : check.ok ? "可用" : "需要检查";
          return (
            <li className={check?.ok ? "check-row ok" : "check-row"} key={key}>
              <span className="check-name">{label}</span>
              <span className="check-msg">{state}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
