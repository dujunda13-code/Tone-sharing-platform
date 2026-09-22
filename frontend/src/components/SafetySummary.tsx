import type { SynthesisSummary, WatermarkResponse } from "../api/types";
import { api } from "../api/client";
import { Disclosure } from "./Disclosure";
import { StatusBadge } from "./StatusBadge";

function formatProbability(value: number | null): string {
  return value === null ? "未复检" : `${(value * 100).toFixed(1)}%`;
}

function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN", { hour12: false });
}

/** 安全记录行：水印、指纹与可追溯结论；结论只由后端复检结果决定，内部 ID 折叠展示。 */
export function SafetySummary({
  item,
  verification,
  onVerify,
  verifying,
}: {
  item: SynthesisSummary;
  verification?: WatermarkResponse;
  onVerify: (jobId: string) => void;
  verifying: boolean;
}) {
  const watermarkOk = item.watermark_probability !== null && item.watermark_probability >= 0.8 && item.fingerprint_anomaly === false;
  return (
    <section className="card safety-row" aria-label={`合成记录 ${formatTime(item.created_at)}`}>
      <header className="resource-card-head">
        <h3>合成记录 · {formatTime(item.created_at)}</h3>
        <StatusBadge status={item.status} />
      </header>
      <dl className="kv">
        <dt>文本语言</dt>
        <dd>{item.text_lang === "zh" ? "中文" : item.text_lang === "en" ? "English" : "—"}</dd>
        <dt>来源水印</dt>
        <dd>{formatProbability(item.watermark_probability)}</dd>
        <dt>声音指纹</dt>
        <dd>{item.fingerprint_anomaly === null ? "未记录" : item.fingerprint_anomaly ? "存在异常" : "未见异常"}</dd>
        <dt>结果判定</dt>
        <dd>{watermarkOk ? "水印校验通过，可安全分发" : "待复检或未通过"}</dd>
      </dl>
      {verification && (
        <p className={verification.payload_matches_job ? "note" : "note warn-note"}>
          {verification.payload_matches_job
            ? "复检结论：可追溯，确认为平台生成。"
            : "复检结论：无法确认来源。严重外部加噪、重编码或处理后可能无法识别，界面不会将其标记为平台生成。"}
        </p>
      )}
      <div className="resource-card-actions">
        <button className="secondary" type="button" disabled={verifying} onClick={() => onVerify(item.job_id)}>
          {verifying ? "复检中…" : "复检来源"}
        </button>
        {item.download_ready && (
          <a className="download-link" role="button" href={api.audioUrl(item.job_id)} download>
            下载音频
          </a>
        )}
      </div>
      <Disclosure label="查看详情">
        任务 ID：{item.job_id}
        {item.voice_profile_id ? ` · 音色档案：${item.voice_profile_id}` : ""}
      </Disclosure>
    </section>
  );
}
