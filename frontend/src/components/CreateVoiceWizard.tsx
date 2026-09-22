import { useEffect, useState } from "react";

import { api, getApiError } from "../api/client";
import type { HealthResponse } from "../api/types";
import { createVoiceDraftStore, useDraft } from "../state/drafts";
import type { CreateVoiceReviewDraft } from "../state/drafts";
import { EMOTION_LABEL_TEXT } from "./VoicePicker";
import { Disclosure } from "./Disclosure";
import type { SafeError } from "./ErrorPanel";
import { ErrorPanel } from "./ErrorPanel";
import { QualityWarning } from "./QualityWarning";
import { StatusBadge } from "./StatusBadge";
import type { WorkspacePage } from "./AppShell";
import { WizardAddReferenceModal } from "./WizardAddReferenceModal";

export type CreateVoiceStep = "upload" | "inspect" | "prepare" | "review" | "saved";

const STEP_LABELS: Array<{ id: CreateVoiceStep; label: string }> = [
  { id: "upload", label: "命名与上传" },
  { id: "inspect", label: "音频分析" },
  { id: "review", label: "内容审核" },
  { id: "prepare", label: "确认档案" },
  { id: "saved", label: "创建完成" },
];

const EMOTION_OPTIONS = Object.keys(EMOTION_LABEL_TEXT);

/**
 * 快速克隆向导：命名并上传已授权的 3–10 秒参考音频 → 音频分析 →
 * 内容审核 → 确认档案 → 保存 ready 零样本音色档案（基础模型合成，无训练）。
 * 所有状态来自真实接口；前置条件不满足时保存保持禁用并显示真实原因。
 * 进度类状态保存在会话草稿中，切换页面后返回可继续上次的操作。
 */
export function CreateVoiceWizard({
  onNavigate,
}: {
  onNavigate: (page: WorkspacePage) => void;
}) {
  const [draft, setDraft] = useDraft(createVoiceDraftStore);
  const { step, displayName, consent, file, dataset, preparedSeconds, segments, review } = draft;
  const pending = draft.pending;
  const error = draft.error;
  const busy = pending !== null;
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [savedProfile, setSavedProfile] = useState<{ id: string; base_model_id: string | null } | null>(null);
  const [showAddAux, setShowAddAux] = useState(false);

  useEffect(() => {
    let active = true;
    void api
      .health()
      .then((next) => {
        if (active) setHealth(next);
      })
      .catch(() => {
        if (active) setHealth(null);
      });
    return () => {
      active = false;
    };
  }, []);

  const upload = async () => {
    if (!file || !consent) return;
    setDraft({ pending: "upload", error: null });
    try {
      const uploaded = await api.uploadDataset(file, true);
      const list = await api.datasets();
      const summary = list.items.find((item) => item.id === uploaded.dataset_id) ?? null;
      setDraft({
        dataset: summary ?? {
          id: uploaded.dataset_id,
          authorization_confirmed: true,
          status: "uploaded",
          effective_seconds: uploaded.effective_seconds ?? null,
          asset_count: 1,
          created_at: new Date().toISOString(),
        },
        step: "inspect",
      });
    } catch (requestError) {
      setDraft({ error: getApiError(requestError) });
    } finally {
      setDraft({ pending: null });
    }
  };

  const prepare = async () => {
    if (!dataset) return;
    setDraft({ pending: "prepare", error: null });
    try {
      const result = await api.preprocess(dataset.id);
      setDraft({
        preparedSeconds: result.effective_seconds,
        dataset: {
          ...dataset,
          status: result.status,
          effective_seconds: result.effective_seconds,
        },
        step: "review",
      });
      const list = await api.segments(dataset.id);
      setDraft({ segments: list });
    } catch (requestError) {
      setDraft({ error: getApiError(requestError) });
    } finally {
      setDraft({ pending: null });
    }
  };

  const refreshSegments = async () => {
    if (!dataset) return;
    const list = await api.segments(dataset.id);
    setDraft({
      segments: list,
      dataset: { ...dataset, status: list.status, effective_seconds: list.effective_seconds },
      // 审核数据刷新后显式进入下一步：审核完成即到确认档案，否则留在内容审核。
      step: list.ready_for_profile && step === "review" ? "prepare" : step,
    });
  };

  const confirmAll = async () => {
    if (!dataset) return;
    setDraft({ pending: "confirm", error: null });
    try {
      await api.confirmSegments(dataset.id, "确认自动转写与自动情绪标签");
      await refreshSegments();
    } catch (requestError) {
      setDraft({ error: getApiError(requestError) });
    } finally {
      setDraft({ pending: null });
    }
  };

  const saveProfile = async () => {
    if (!dataset || !savable) return;
    setDraft({ pending: "save", error: null });
    try {
      const profile = await api.createVoice(dataset.id, displayName);
      // 档案创建后逐个挂接向导内已审核的辅助参考；任一失败即终止并保留错误，
      // 主档案已建立，不回滚——用户可在音色库继续管理辅助参考。
      for (const aux of draft.auxiliaryReferences) {
        await api.addVoiceReference(profile.id, aux.datasetId);
      }
      setSavedProfile({ id: profile.id, base_model_id: profile.base_model_id ?? null });
      setDraft({ step: "saved" });
    } catch (requestError) {
      setDraft({ error: getApiError(requestError) });
    } finally {
      setDraft({ pending: null });
    }
  };

  const effectiveSeconds = preparedSeconds ?? dataset?.effective_seconds ?? null;
  const serviceReady = health?.status === "ready";
  const confirmed = dataset?.authorization_confirmed === true;
  const reviewed = segments !== null && segments.ready_for_profile;
  const durationOk = effectiveSeconds !== null && effectiveSeconds >= 3 && effectiveSeconds <= 10;
  const nameOk = displayName.trim() !== "";
  const savable = confirmed && reviewed && durationOk && serviceReady && nameOk;

  const blockers: string[] = [];
  if (!nameOk) blockers.push("请先为音色命名");
  if (!confirmed) blockers.push("参考音频未完成授权确认");
  if (segments !== null && segments.reviewed_segments < segments.total_segments) {
    blockers.push("仍有片段未完成文本/情绪审核");
  }
  if (!reviewed) blockers.push("数据集未完成审核确认");
  if (!durationOk) blockers.push("有效时长需在 3–10 秒");
  if (!serviceReady) blockers.push("服务未就绪（待准备）");

  return (
    <div className="wizard preview-wizard">
      <ol className="wizard-steps" aria-label="创建音色步骤">
        {STEP_LABELS.map((item, index) => (
          <li
            key={item.id}
            className={
              STEP_LABELS.findIndex((s) => s.id === step) >= index ? "done" : ""
            }
            aria-current={step === item.id ? "step" : undefined}
          >
            {item.label}
          </li>
        ))}
      </ol>

      <div className="create-upload-stage">
      <section className="card create-upload-panel">
        <header className="resource-card-head">
          <h2>快速克隆音色</h2>
          {dataset && <StatusBadge status={dataset.status} />}
        </header>
        <p className="card-lede">
          上传一段已授权的 3–10 秒参考音频（推荐 5–8 秒），系统完成质检、转写与情绪审核后，
          立即生成音色档案；合成直接使用冻结的基础模型权重，全程不做训练。
        </p>
        <div className="field">
          <label htmlFor="wizard-voice-name">音色名称</label>
          <input
            id="wizard-voice-name"
            type="text"
            value={displayName}
            maxLength={64}
            disabled={step === "saved"}
            placeholder="例如：温柔旁白、产品讲解"
            onChange={(event) => setDraft({ displayName: event.target.value })}
          />
          <p className="hint">1–64 个字符，创建后用于在音色库和语音创作中展示。</p>
        </div>
        <label className="consent-box">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setDraft({ consent: event.target.checked })}
            disabled={step !== "upload"}
          />
          <span>我确认拥有该声音的参考音频授权</span>
        </label>
        <div className="field">
          <label className="file-dropzone" htmlFor="wizard-file">
            <span className="dropzone-icon" aria-hidden="true">↑</span>
            <span className="dropzone-title">授权音频文件</span>
            <span className="dropzone-copy">点击上传或将文件拖放到这里</span>
            <span className="dropzone-meta">支持 WAV、FLAC、MP3 · 建议 3–10 秒</span>
            <input
              id="wizard-file"
              aria-label="授权音频文件"
              className="file-input-hidden"
              type="file"
              accept=".wav,.flac,.mp3,audio/*"
              disabled={step !== "upload"}
              onChange={(event) => setDraft({ file: event.target.files?.[0] ?? null })}
            />
          </label>
          {file && <p className="hint">已选择：{file.name}（切换页面后仍保留）</p>}
          <p className="hint">参考音频 3–10 秒，推荐 5–8 秒；格式 WAV/FLAC/MP3。</p>
        </div>
        <button
          className="primary"
          type="button"
          disabled={step !== "upload" || !consent || !file || busy}
          onClick={() => void upload()}
        >
          {pending === "upload" ? "上传中…" : "上传并分析"}
        </button>
        {step !== "saved" && (
          <div className="create-aux-panel" aria-label="辅助参考音频">
            <header className="resource-card-head">
              <h3>辅助参考音频（{draft.auxiliaryReferences.length}/4）</h3>
              {draft.auxiliaryReferences.length < 4 && (
                <button className="secondary" type="button" disabled={busy} onClick={() => setShowAddAux(true)}>
                  添加辅助参考
                </button>
              )}
            </header>
            <p className="card-lede">可选：追加最多 4 段已授权的 3–10 秒参考，保存音色时一并挂接。</p>
            {draft.auxiliaryReferences.length > 0 && (
              <ul className="aux-reference-list">
                {draft.auxiliaryReferences.map((ref) => (
                  <li key={ref.id} className="aux-reference-row">
                    <div className="aux-reference-meta">
                      <span className="aux-reference-name">{ref.fileName}</span>
                      <span className="aux-reference-detail">
                        {ref.effectiveSeconds.toFixed(1)} 秒 · {EMOTION_LABEL_TEXT[ref.emotionLabel as keyof typeof EMOTION_LABEL_TEXT] ?? ref.emotionLabel}
                      </span>
                    </div>
                    <button className="secondary" type="button" disabled={busy} onClick={() => setDraft({ auxiliaryReferences: draft.auxiliaryReferences.filter((r) => r.id !== ref.id) })}>
                      移除
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        {error && <ErrorPanel error={error} />}
      </section>
      <aside className="card create-audio-preview" aria-label="音频预览">
        <div className="preview-card-heading">
          <div>
            <h2>音频预览</h2>
            <p>上传后将显示参考声音的时长与质量提示。</p>
          </div>
          <span className="preview-status-dot" aria-hidden="true" />
        </div>
        <div className="preview-waveform" aria-hidden="true">
          {Array.from({ length: 34 }, (_, index) => (
            <span key={index} style={{ height: `${25 + ((index * 19) % 58)}%` }} />
          ))}
        </div>
        <div className="preview-audio-meta">
          <span>{file ? file.name : "等待选择参考音频"}</span>
          <span>{file ? "已就绪" : "00:00 / --:--"}</span>
        </div>
        <div className="upload-tips" aria-label="上传提示">
          <h3>上传提示</h3>
          <ul>
            <li>清晰录制、单人发声</li>
            <li>建议安静环境，避免背景音乐</li>
            <li>支持 WAV、FLAC、MP3 格式</li>
          </ul>
        </div>
      </aside>
      </div>

      {dataset && (
        <section className="card">
          <header className="resource-card-head">
            <h2>音频分析</h2>
            <button className="secondary" type="button" disabled={step !== "inspect" || busy} onClick={() => void prepare()}>
              {pending === "prepare" ? "质检中…" : "开始质检与准备"}
            </button>
          </header>
          {pending === "prepare" && (
            <p className="note" role="status">
              质检进行中：质检与转写需要片刻，期间可前往其他页面；完成后回到这里会显示待审核的参考内容。
            </p>
          )}
          <dl className="kv">
            <dt>授权确认</dt>
            <dd>{dataset.authorization_confirmed ? "已确认" : "未确认"}</dd>
            <dt>有效语音时长</dt>
            <dd>{effectiveSeconds !== null ? `${effectiveSeconds} 秒` : "质检后生成"}</dd>
            <dt>质检状态</dt>
            <dd>
              {dataset.status === "review_required"
                ? "转写完成，待审核"
                : dataset.status === "ready_for_profile"
                  ? "审核完成，可保存音色档案"
                  : "待质检"}
            </dd>
          </dl>
          <Disclosure label="技术详情">数据集 ID：{dataset.id}</Disclosure>
          <p className="note">
            质检、转写与自动情绪识别在系统内完成；参考内容确认后即可保存零样本音色档案。
          </p>
        </section>
      )}

      {segments && (step === "review" || step === "prepare" || step === "saved") && (
        <section className="card" aria-label="内容审核">
          <header className="resource-card-head">
            <h2>内容审核</h2>
            <span>
              已审核 {segments.reviewed_segments} / {segments.total_segments} 段
            </span>
            {segments.status === "review_required" && (
              <button className="secondary" type="button" disabled={busy} onClick={() => void confirmAll()}>
                全部确认自动结果
              </button>
            )}
          </header>
          <p className="card-lede">
            逐段校对自动转写文本、必要时修正情绪标签；人工修改不会覆盖自动记录。
            只有审核完成且文本非空的参考片段才能保存音色档案。
          </p>
          <ul className="segment-list">
            {segments.items.map((item) => (
              <li key={item.segment_id} className="segment-row">
                <div className="segment-head">
                  <code>{item.segment_id}</code>
                  <span>{item.duration_seconds} 秒</span>
                  <span className={`segment-state ${item.reviewed ? "ok" : "pending"}`}>
                    {item.reviewed ? "已审核" : "待审核"}
                  </span>
                </div>
                <QualityWarning
                  warningCodes={item.warning_codes}
                  snrDb={item.snr_db}
                  thresholdDb={item.snr_warning_db}
                  compact
                />
                <p>
                  有效文本：{item.effective_transcript || "（空，需人工补充）"}
                  <span className="hint">（来源：{item.transcript_source === "manual" ? "人工校对" : "自动转写"}）</span>
                </p>
                <p>
                  有效情绪：{item.effective_emotion_label ? EMOTION_LABEL_TEXT[item.effective_emotion_label] ?? item.effective_emotion_label : "未识别"}
                  <span className="hint">
                    （自动：{item.auto_emotion_label ? EMOTION_LABEL_TEXT[item.auto_emotion_label] ?? item.auto_emotion_label : "无"}
                    {item.auto_emotion_confidence !== null ? `，置信度 ${item.auto_emotion_confidence.toFixed(2)}` : ""}）
                  </span>
                </p>
              </li>
            ))}
          </ul>
          <SegmentReviewEditor
            datasetId={dataset?.id ?? ""}
            disabled={busy || segments.status !== "review_required"}
            draft={review}
            onDraftChange={(patch) => setDraft({ review: { ...review, ...patch } })}
            onSaved={() => void refreshSegments()}
          />
        </section>
      )}

      {dataset && (step === "prepare" || step === "saved") && (
        <section className="card" aria-label="确认档案">
          <header className="resource-card-head">
            <h2>确认档案</h2>
          </header>
          <p className="card-lede">
            保存后音色档案立即可用：合成将直接使用这条参考音频与系统内置的基础声音模型，全程无需训练。
          </p>
          {blockers.length > 0 ? (
            <ul className="blocker-list">
              {blockers.map((reason) => (
                <li key={reason}>待准备：{reason}</li>
              ))}
            </ul>
          ) : (
            <p className="note">所有前置条件已满足，可以保存零样本音色档案。</p>
          )}
          <button
            className="primary"
            type="button"
            disabled={!savable || busy || step === "saved"}
            onClick={() => void saveProfile()}
          >
            {pending === "save" ? "保存中…" : "保存音色档案"}
          </button>
        </section>
      )}

      {step === "saved" && savedProfile && (
        <section className="card" aria-label="创建完成">
          <p className="eyebrow">创建完成</p>
          <h2>「{displayName.trim() || "音色"}」创建完成</h2>
          <p className="card-lede">零样本档案已就绪，可以立即用这个声音进行语音创作。</p>
          {draft.auxiliaryReferences.length > 0 && (
            <p className="note">
              已附加 {draft.auxiliaryReferences.length} 段辅助参考音频，可在音色库继续管理。
            </p>
          )}
          <Disclosure label="技术详情">
            档案 ID：{savedProfile.id}
            {savedProfile.base_model_id ? ` · 基础模型：${savedProfile.base_model_id}` : ""}
          </Disclosure>
          <div className="dash-actions">
            <button className="primary" type="button" onClick={() => onNavigate("synthesize")}>
              立即合成
            </button>
            <button className="secondary" type="button" onClick={() => onNavigate("voices")}>
              查看我的音色
            </button>
          </div>
        </section>
      )}

      {showAddAux && (
        <WizardAddReferenceModal
          onClose={() => setShowAddAux(false)}
          onAdd={(ref) => {
            setDraft({ auxiliaryReferences: [...draft.auxiliaryReferences, ref] });
            setShowAddAux(false);
          }}
        />
      )}
    </div>
  );
}

function SegmentReviewEditor({
  datasetId,
  disabled,
  draft,
  onDraftChange,
  onSaved,
}: {
  datasetId: string;
  disabled: boolean;
  draft: CreateVoiceReviewDraft;
  onDraftChange: (patch: Partial<CreateVoiceReviewDraft>) => void;
  onSaved: () => void;
}) {
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<SafeError | null>(null);

  const save = async () => {
    if (!draft.segmentId || !draft.reason.trim()) return;
    setError(null);
    setSaved(false);
    try {
      await api.reviewSegment(datasetId, draft.segmentId, {
        transcript: draft.transcript.trim() ? draft.transcript.trim() : undefined,
        emotion_label: draft.emotion || undefined,
        language:
          draft.language === "zh" || draft.language === "en" ? draft.language : undefined,
        reason: draft.reason.trim(),
      });
      setSaved(true);
      onSaved();
    } catch (requestError) {
      setError(getApiError(requestError));
    }
  };

  return (
    <div className="segment-editor">
      <h3>人工校对片段</h3>
      <div className="field">
        <label htmlFor="review-segment-id">分段 ID</label>
        <input
          id="review-segment-id"
          value={draft.segmentId}
          disabled={disabled}
          onChange={(event) => {
            onDraftChange({ segmentId: event.target.value });
            setSaved(false);
          }}
          placeholder="例如 seg_0001"
        />
      </div>
      <div className="field">
        <label htmlFor="review-transcript">校对后文本（留空则仅修改情绪）</label>
        <textarea
          id="review-transcript"
          value={draft.transcript}
          disabled={disabled}
          rows={2}
          onChange={(event) => onDraftChange({ transcript: event.target.value })}
        />
      </div>
      <div className="field">
        <label htmlFor="review-emotion">修正情绪标签（可选）</label>
        <select
          id="review-emotion"
          value={draft.emotion}
          disabled={disabled}
          onChange={(event) => onDraftChange({ emotion: event.target.value })}
        >
          <option value="">保持自动结果</option>
          {EMOTION_OPTIONS.map((label) => (
            <option key={label} value={label}>
              {EMOTION_LABEL_TEXT[label] ?? label}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor="review-language">参考语言（可选）</label>
        <select
          id="review-language"
          value={draft.language}
          disabled={disabled}
          onChange={(event) => onDraftChange({ language: event.target.value })}
        >
          <option value="">保持自动结果</option>
          <option value="zh">中文</option>
          <option value="en">English</option>
        </select>
      </div>
      <div className="field">
        <label htmlFor="review-reason">修改理由（必填）</label>
        <input
          id="review-reason"
          value={draft.reason}
          disabled={disabled}
          onChange={(event) => onDraftChange({ reason: event.target.value })}
        />
      </div>
      <button
        className="secondary"
        type="button"
        disabled={disabled || !draft.segmentId || !draft.reason.trim()}
        onClick={() => void save()}
      >
        保存审核结果
      </button>
      {saved && <p className="note">审核结果已保存。</p>}
      {error && <ErrorPanel error={error} />}
    </div>
  );
}
