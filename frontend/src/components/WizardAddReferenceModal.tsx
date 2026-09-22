import { useState } from "react";
import { api, getApiError } from "../api/client";
import type { SafeError } from "./ErrorPanel";
import { ErrorPanel } from "./ErrorPanel";
import { EMOTION_LABEL_TEXT } from "./VoicePicker";
import type { WizardAuxiliaryReference } from "../state/drafts";

export function WizardAddReferenceModal({
  onClose,
  onAdd,
}: {
  onClose: () => void;
  onAdd: (ref: WizardAuxiliaryReference) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [consent, setConsent] = useState(false);
  const [stage, setStage] = useState<"idle" | "uploading" | "preprocessing" | "reviewing" | "confirming">("idle");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [segmentId, setSegmentId] = useState<string | null>(null);
  const [effectiveSeconds, setEffectiveSeconds] = useState<number>(0);
  const [promptText, setPromptText] = useState("");
  const [emotionLabel, setEmotionLabel] = useState("neutral");
  const [language, setLanguage] = useState<"zh" | "en">("zh");
  const [error, setError] = useState<SafeError | null>(null);

  const handleUploadAndAnalyze = async () => {
    if (!file || !consent) return;
    setError(null);
    setStage("uploading");
    try {
      const uploadRes = await api.uploadDataset(file, true);
      const dsId = uploadRes.dataset_id;
      setDatasetId(dsId);

      setStage("preprocessing");
      const prepRes = await api.preprocess(dsId);
      setEffectiveSeconds(prepRes.effective_seconds || 5.0);

      const segList = await api.segments(dsId);
      if (segList.items.length === 0) {
        throw new Error("未能从音频中提取到有效的 3–10 秒人声音频");
      }
      const seg = segList.items[0];
      setSegmentId(seg.segment_id);
      setPromptText(seg.effective_transcript || "");
      setEmotionLabel(seg.effective_emotion_label || "neutral");
      setLanguage(seg.language === "en" ? "en" : "zh");
      setStage("reviewing");
    } catch (err) {
      setError(getApiError(err));
      setStage("idle");
    }
  };

  const handleConfirmAndAdd = async () => {
    if (!datasetId || !segmentId || !promptText.trim()) return;
    setError(null);
    setStage("confirming");
    try {
      await api.reviewSegment(datasetId, segmentId, {
        transcript: promptText.trim(),
        emotion_label: emotionLabel,
        language: language,
        reason: "创建向导中确认辅助参考音频",
      });

      await api.confirmSegments(datasetId, "确认辅助参考音频");

      onAdd({
        id: `aux-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        datasetId,
        effectiveSeconds,
        emotionLabel,
        transcript: promptText.trim(),
        fileName: file?.name || "audio.wav",
      });
      onClose();
    } catch (err) {
      setError(getApiError(err));
      setStage("reviewing");
    }
  };

  return (
    <div
      className="modal-backdrop"
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.4)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        className="card modal-content"
        style={{
          width: "min(560px, calc(100vw - 32px))",
          maxHeight: "90vh",
          overflowY: "auto",
          backgroundColor: "#ffffff",
          borderRadius: "12px",
          padding: "24px",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ margin: 0, fontSize: "20px" }}>添加辅助参考音频</h2>
          <button type="button" className="secondary" style={{ minHeight: "auto", padding: "4px 8px" }} onClick={onClose}>
            ✕
          </button>
        </div>

        {error && <ErrorPanel error={error} />}

        {stage === "idle" && (
          <div className="form-stack" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <label style={{ display: "block", marginBottom: 6, fontSize: "14px", fontWeight: 500 }}>
                选择音频文件 (3–10 秒，WAV/FLAC/MP3)
              </label>
              <input
                type="file"
                accept=".wav,.flac,.mp3"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>

            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "13px" }}>
              <input
                type="checkbox"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
              />
              <span>我确认已获得该音频的合法授权</span>
            </label>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              <button type="button" className="secondary" onClick={onClose}>取消</button>
              <button
                type="button"
                className="primary"
                disabled={!file || !consent}
                onClick={handleUploadAndAnalyze}
              >
                上传并分析
              </button>
            </div>
          </div>
        )}

        {(stage === "uploading" || stage === "preprocessing" || stage === "confirming") && (
          <div style={{ padding: "32px 0", textAlign: "center" }}>
            <p style={{ fontSize: "16px", color: "var(--ink)" }}>
              {stage === "uploading" && "正在上传音频…"}
              {stage === "preprocessing" && "正在执行 VAD 截取、信噪比质检与语音转写…"}
              {stage === "confirming" && "正在确认审核结果…"}
            </p>
          </div>
        )}

        {stage === "reviewing" && (
          <div className="form-stack" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <label htmlFor="wiz-aux-ref-text" style={{ display: "block", marginBottom: 6, fontSize: "14px", fontWeight: 500 }}>
                参考文本 (请核对并修正)
              </label>
              <textarea
                id="wiz-aux-ref-text"
                rows={3}
                style={{ width: "100%", padding: "8px", borderRadius: "6px", border: "1px solid var(--hairline-strong)" }}
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
              />
            </div>

            <div style={{ display: "flex", gap: 16 }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: "block", marginBottom: 6, fontSize: "14px", fontWeight: 500 }}>
                  情绪标签
                </label>
                <select
                  style={{ width: "100%", padding: "8px", borderRadius: "6px", border: "1px solid var(--hairline-strong)" }}
                  value={emotionLabel}
                  onChange={(e) => setEmotionLabel(e.target.value)}
                >
                  {Object.entries(EMOTION_LABEL_TEXT).map(([key, label]) => (
                    <option key={key} value={key}>{label}</option>
                  ))}
                </select>
              </div>
              <div style={{ flex: 1 }}>
                <label style={{ display: "block", marginBottom: 6, fontSize: "14px", fontWeight: 500 }}>
                  语言
                </label>
                <select
                  style={{ width: "100%", padding: "8px", borderRadius: "6px", border: "1px solid var(--hairline-strong)" }}
                  value={language}
                  onChange={(e) => setLanguage(e.target.value as "zh" | "en")}
                >
                  <option value="zh">中文</option>
                  <option value="en">英文</option>
                </select>
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
              <button type="button" className="secondary" onClick={() => setStage("idle")}>重新上传</button>
              <button
                type="button"
                className="primary"
                disabled={!promptText.trim()}
                onClick={handleConfirmAndAdd}
              >
                确认添加该参考
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
