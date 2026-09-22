import { Disclosure } from "./Disclosure";
import { EMOTION_LABEL_TEXT } from "./VoicePicker";
import { QualityWarning } from "./QualityWarning";
import { StatusBadge } from "./StatusBadge";
import type { VoiceSummary } from "../api/types";

function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString("zh-CN", { hour12: false });
}

/**
 * 音色行（ElevenLabs voice-row）：圆形声音图标、用户命名、情绪与状态。
 * 档案 ID、数据集 ID 与基础模型 ID 属于技术详情，默认折叠。
 */
export function VoiceCard({
  voice,
  onSynthesize,
  onNavigate,
  onAddReference,
  onDeleteReference,
}: {
  voice: VoiceSummary;
  onSynthesize: (voiceId: string) => void;
  onNavigate: (page: "create" | "tasks") => void;
  onAddReference?: (voice: VoiceSummary) => void;
  onDeleteReference?: (profileId: string, referenceId: string) => void;
}) {
  const emotionText = (voice.reference_emotions ?? [])
    .map((label) => EMOTION_LABEL_TEXT[label] ?? label)
    .join("、");

  const refCount = voice.references?.length ?? (voice.mode === "zero_shot" ? 1 : 0);
  const canAddMore = refCount < 5;

  return (
    <div className="voice-row">
      <div className="voice-row-identity">
        <span className="voice-icon" aria-hidden="true">{voice.display_name.trim().slice(0, 1) || "声"}</span>
        <div className="voice-row-title">{voice.display_name}</div>
      </div>
      <div className="voice-row-tags" aria-label="音色标签">
        <span className="voice-tag">{voice.mode === "zero_shot" ? "零样本" : "历史档案"}</span>
        {(voice.reference_emotions ?? []).map((emotion) => <span className="voice-tag" key={emotion}>{EMOTION_LABEL_TEXT[emotion] ?? emotion}</span>)}
      </div>
      <div className="voice-row-state"><StatusBadge status={voice.status} /></div>
      <time className="voice-row-created" dateTime={voice.created_at}>{formatTime(voice.created_at)}</time>
      <div className="voice-row-actions">
        {canAddMore && onAddReference && voice.status === "ready" && (
          <button className="secondary" type="button" onClick={() => onAddReference(voice)}>＋ 添加参考</button>
        )}
        {voice.can_synthesize && voice.status === "ready" && (
          <button className="primary" type="button" onClick={() => onSynthesize(voice.id)}>用此音色创作</button>
        )}
        {voice.status !== "ready" && (
          <button className="secondary" type="button" onClick={() => onNavigate("tasks")}>查看任务</button>
        )}
      </div>
      <div className="voice-row-details">
        <div className="voice-row-meta">
          <span className="voice-ref-count-badge">参考音频 {refCount}/5</span> · {emotionText ? `参考情绪 ${emotionText} · ` : ""}{voice.can_synthesize && voice.status === "ready" ? "可用于语音创作" : "暂不可合成"}
        </div>
        <QualityWarning
          warningCodes={voice.quality_warning_codes}
          snrDb={voice.reference_snr_db}
          compact
        />
        {voice.references && voice.references.length > 0 && (
          <Disclosure label={`查看参考音频列表 (${voice.references.length}/5)`}>
            <ul className="voice-references-list" style={{ margin: "8px 0", paddingLeft: 16 }}>
              {voice.references.map((ref) => (
                <li key={ref.id} style={{ marginBottom: 4, display: "flex", alignItems: "center", gap: 8 }}>
                  <strong style={{ fontSize: "12px", color: ref.is_primary ? "var(--ink)" : "var(--muted)" }}>
                    {ref.reference_name ? `「${ref.reference_name}」` : (ref.is_primary ? "【主参考】" : "【辅助参考】")}
                  </strong>
                  <span>{ref.prompt_text}</span>
                  <span style={{ color: "var(--muted)", fontSize: "12px" }}>
                    ({EMOTION_LABEL_TEXT[ref.emotion_label] ?? ref.emotion_label})
                  </span>
                  {!ref.is_primary && onDeleteReference && (
                    <button
                      type="button"
                      className="secondary"
                      style={{ padding: "2px 8px", fontSize: "12px", minHeight: "auto" }}
                      onClick={() => onDeleteReference(voice.id, ref.id)}
                    >
                      删除
                    </button>
                  )}
                </li>
              ))}
            </ul>
          </Disclosure>
        )}
        <Disclosure label="技术详情">
          档案 ID：{voice.id} · 数据集 ID：{voice.dataset_id}
          {voice.base_model_id ? ` · 基础模型：${voice.base_model_id}` : ""}
        </Disclosure>
      </div>
    </div>
  );
}
