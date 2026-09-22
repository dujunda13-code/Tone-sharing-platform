import { useEffect, useState } from "react";

import { api, getApiError } from "../api/client";
import type {
  CloudEmotionLabel,
  EmotionControl,
  EmotionLabel,
  JobStatus,
  SynthesisLanguage,
  SynthesisMode,
  SynthesisSummary,
  VoiceSummary,
} from "../api/types";
import { synthesisDraftStore, useDraft } from "../state/drafts";
import type { SynthesisRequest } from "../state/drafts";
import type { WorkspacePage } from "../components/AppShell";
import { AudioResultCard } from "../components/AudioResultCard";
import type { SafeError } from "../components/ErrorPanel";
import { ErrorPanel } from "../components/ErrorPanel";
import { EmptyState } from "../components/EmptyState";
import { JobProgress } from "../components/JobProgress";
import { Dropdown } from "../components/Dropdown";
import { QualityWarning } from "../components/QualityWarning";
import { EMOTION_LABEL_TEXT, VoicePicker } from "../components/VoicePicker";

const TERMINAL = new Set<JobStatus>(["succeeded", "failed", "canceled"]);
const CLOUD_EMOTION_LABELS: Array<{ value: CloudEmotionLabel; label: string }> = [
  { value: "neutral", label: "中性" },
  { value: "happy", label: "开心" },
  { value: "angry", label: "愤怒" },
  { value: "sad", label: "悲伤" },
  { value: "afraid", label: "恐惧" },
  { value: "disgusted", label: "厌恶" },
  { value: "melancholic", label: "忧郁" },
  { value: "surprised", label: "惊讶" },
  { value: "calm", label: "平静" },
];
const VECTOR_LABELS = ["开心", "愤怒", "悲伤", "恐惧", "厌恶", "忧郁", "惊讶", "平静"];

/** 语音创作工作台：从可用音色选择、输入文本并安全合成；只有后端确认的已验证输出才提供播放/下载。 */
export function SynthesisStudioPage({
  synthesisRequest,
  onNavigate,
}: {
  synthesisRequest?: SynthesisRequest | null;
  onNavigate: (page: WorkspacePage) => void;
}) {
  const [draft, setDraft] = useDraft(synthesisDraftStore);
  const {
    selected,
    text,
    language,
    synthesisMode,
    localOptions,
    emotionMode,
    emotionLabel,
    strength,
    consent,
    cloudControlMode,
    cloudEmotionLabel,
    cloudDescription,
    cloudVector,
    cloudVectorMode,
    cloudSampleRate,
    cloudSpeed,
    cloudGain,
    cloudUseRandom,
    cloudIntervalSilence,
    cloudConsent,
    jobId,
  } = draft;
  const [voices, setVoices] = useState<VoiceSummary[] | null>(null);
  const [loadError, setLoadError] = useState<SafeError | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [synthesis, setSynthesis] = useState<SynthesisSummary | null>(null);
  const [error, setError] = useState<SafeError | null>(null);
  const [plazaVoice, setPlazaVoice] = useState<{ id: string; label: string } | null>(null);

  useEffect(() => {
    let active = true;
    void api
      .voices()
      .then((list) => {
        if (!active) return;
        setVoices(list.items);
        setLoadError(null);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setVoices(null);
        setLoadError(getApiError(requestError));
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (voices === null || !synthesisRequest) return;
    const selectedVoiceIsReady = voices.some(
      (voice) => voice.id === draft.selected && voice.can_synthesize && voice.status === "ready",
    );
    const selectedVoiceIsRequestedPlazaVoice =
      synthesisRequest.voiceLabel !== undefined && draft.selected === synthesisRequest.voiceId;
    if (
      synthesisRequest.token === draft.appliedRequestToken &&
      (selectedVoiceIsReady || selectedVoiceIsRequestedPlazaVoice)
    ) {
      return;
    }
    const match = voices.find((voice) => voice.id === synthesisRequest.voiceId);
    if (match?.can_synthesize && match.status === "ready") {
      setPlazaVoice(null);
      setDraft({ selected: synthesisRequest.voiceId, appliedRequestToken: synthesisRequest.token });
    } else if (synthesisRequest.voiceLabel) {
      setPlazaVoice({ id: synthesisRequest.voiceId, label: synthesisRequest.voiceLabel });
      setDraft({ selected: synthesisRequest.voiceId, appliedRequestToken: synthesisRequest.token });
    }
  }, [voices, synthesisRequest, draft.appliedRequestToken, draft.selected, setDraft]);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const list = await api.synthesisList();
        if (!active) return;
        const next = list.items.find((item) => item.job_id === jobId) ?? null;
        setSynthesis(next);
        if (next && TERMINAL.has(next.status) && timer !== undefined) {
          window.clearInterval(timer);
        }
      } catch {
        /* 轮询失败时保留上一次真实状态，不伪造 */
      }
    };
    void poll();
    timer = window.setInterval(() => void poll(), 1000);
    return () => {
      active = false;
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [jobId]);

  const readyVoices = (voices ?? []).filter((voice) => voice.can_synthesize && voice.status === "ready");
  const plazaOption: VoiceSummary | null = plazaVoice ? { id: plazaVoice.id, display_name: plazaVoice.label, dataset_id: "plaza", status: "ready", created_at: "", can_synthesize: true, mode: "zero_shot", base_model_id: null, reference_emotions: [], reference_snr_db: null, quality_warning_codes: [] } : null;
  const pickerVoices = plazaOption && !readyVoices.some((voice) => voice.id === plazaOption.id) ? [...readyVoices, plazaOption] : readyVoices;
  const selectedVoice = pickerVoices.find((voice) => voice.id === selected) ?? null;
  const availableEmotions = selectedVoice?.reference_emotions ?? [];
  const manualOptions = Object.keys(EMOTION_LABEL_TEXT).filter((label) => availableEmotions.includes(label));
  const emotionReferenceUnavailable = synthesisMode === "local" &&
    emotionMode === "manual" && !availableEmotions.includes(emotionLabel);
  const cloudVectorTotal = cloudVector.reduce((sum, value) => sum + value, 0);
  const cloudVectorNonZero = cloudVector.filter((value) => value > 0).length;
  const cloudVectorValid = cloudControlMode !== "vector" || (
    cloudVectorTotal > 0 && cloudVectorTotal <= 1.5 &&
    (cloudVectorMode === "single" ? cloudVectorNonZero === 1 : cloudVectorNonZero >= 2)
  );
  const cloudConfigurationComplete = synthesisMode !== "emotion_api" || (
    cloudConsent &&
    text.length <= 600 &&
    (cloudControlMode !== "description" || cloudDescription.trim() !== "") &&
    cloudVectorValid
  );
  const canSubmit =
    selected !== "" && text.trim() !== "" && consent && !submitting &&
    !emotionReferenceUnavailable && cloudConfigurationComplete;

  const selectSynthesisMode = (mode: SynthesisMode) => {
    setDraft({
      synthesisMode: mode,
      language: mode === "local" && !["zh", "en"].includes(language) ? "zh" : language,
    });
  };

  const submit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const emotion: EmotionControl =
        synthesisMode === "local" && emotionMode === "manual"
          ? { mode: "manual", label: emotionLabel, strength }
          : { mode: "auto", strength };
      const queued = await api.synthesize({
        voice_profile_id: selected,
        text,
        text_lang: language,
        emotion,
        consent_confirmed: true,
        synthesis_mode: synthesisMode,
        local_options: localOptions,
        cloud_processing_confirmed: synthesisMode === "emotion_api" && cloudConsent,
        cloud_options: synthesisMode === "emotion_api" ? {
          control_mode: cloudControlMode,
          label: cloudControlMode === "label" ? cloudEmotionLabel : undefined,
          description: cloudControlMode === "description" ? cloudDescription.trim() : undefined,
          emotion_strength: strength,
          sample_rate: cloudSampleRate,
          speed: cloudSpeed,
          gain: cloudGain,
          use_random: cloudUseRandom,
          interval_silence: cloudIntervalSilence,
          emotion_vector: cloudControlMode === "vector" ? cloudVector : undefined,
          emotion_vector_mode: cloudControlMode === "vector" ? cloudVectorMode : undefined,
        } : undefined,
      });
      setDraft({ jobId: queued.job_id });
      setSynthesis(null);
    } catch (requestError) {
      setError(getApiError(requestError));
    } finally {
      setSubmitting(false);
    }
  };

  if (voices === null && loadError === null) {
    return (
      <section className="card" aria-live="polite">
        <h2>正在加载可用音色…</h2>
      </section>
    );
  }

  if (loadError) {
    return (
      <section className="card">
        <h2>语音创作</h2>
        <ErrorPanel error={loadError} hint="音色列表接口由后端提供；未实现前保持失败提示。" />
      </section>
    );
  }

  if (voices !== null && pickerVoices.length === 0) {
    return (
      <section className="card">
        <h2>语音创作</h2>
        <EmptyState
          title="暂无可用音色"
          description="只有参考与基础模型均就绪的零样本音色才能用于合成。"
          action={
            <button className="primary" type="button" onClick={() => onNavigate("create")}>
              去创建音色
            </button>
          }
        />
      </section>
    );
  }

  return (
    <div className="page-stack preview-page synthesis-page">
      <section className="card page-intro">
        <h2>语音创作</h2>
        <p className="card-lede">
          选择可用音色并输入文本；提交后系统正在生成并完成质量检查，通过后即可播放或下载。
        </p>
      </section>
      <div className="studio-layout synthesis-console-grid">
        <section className="card synthesis-writing-panel">
          <VoicePicker voices={pickerVoices} value={selected} onChange={(voiceId) => setDraft({ selected: voiceId })} />
          {selectedVoice && (
            <QualityWarning
              warningCodes={selectedVoice.quality_warning_codes}
              snrDb={selectedVoice.reference_snr_db}
            />
          )}
          <div className="field synthesis-text-field">
            <label htmlFor="synthesis-text">合成文本</label>
            <textarea
              id="synthesis-text"
              value={text}
              onChange={(event) => setDraft({ text: event.target.value })}
              maxLength={synthesisMode === "emotion_api" ? 600 : 1000}
              placeholder={`输入要合成的文本内容（最多 ${synthesisMode === "emotion_api" ? 600 : 1000} 字）`}
            />
            <p className="hint">{text.length}/{synthesisMode === "emotion_api" ? 600 : 1000} · 提交前后端敏感词过滤是唯一权威判断。</p>
          </div>
        </section>
        <section className="card synthesis-settings-panel">
          <div className="settings-card-heading">
            <div>
              <h3>合成设置</h3>
              <p>调整语言与声音表达方式。</p>
            </div>
          </div>
          <div className="synthesis-mode-switch" role="group" aria-label="合成方式">
            <button
              type="button"
              className={synthesisMode === "local" ? "active" : ""}
              aria-pressed={synthesisMode === "local"}
              onClick={() => selectSynthesisMode("local")}
            >
              正常合成
            </button>
            <button
              type="button"
              className={synthesisMode === "emotion_api" ? "active" : ""}
              aria-pressed={synthesisMode === "emotion_api"}
              onClick={() => selectSynthesisMode("emotion_api")}
            >
              情绪合成
            </button>
          </div>
          <p className="hint mode-explanation">
            {synthesisMode === "local"
              ? "使用设备内 GPT-SoVITS V2ProPlus，不上传文本或参考音频。"
              : "调用云端 ModelVerse IndexTTS-2 情绪合成；参考音频需为 5–30 秒 WAV/MP3，提交前需要单独确认云端处理。"}
          </p>
          <div className="field">
            <label>文本语言</label>
            <Dropdown
              label="文本语言"
              value={language}
              onChange={(value) => setDraft({ language: value as SynthesisLanguage })}
              options={[
                { value: "zh", label: "中文" },
                { value: "en", label: "English" },
                ...(synthesisMode === "emotion_api"
                  ? [
                      { value: "ja", label: "日本語" },
                      { value: "es", label: "Español" },
                      { value: "ar", label: "العربية" },
                    ]
                  : []),
              ]}
            />
          </div>
          {synthesisMode === "local" ? (
            <>
              <div className="field settings-grid">
                <label htmlFor="local-cut-method">文本切分方式</label>
                <select id="local-cut-method" value={localOptions.cut_method} onChange={(event) => setDraft({ localOptions: { ...localOptions, cut_method: event.target.value as typeof localOptions.cut_method } })}>
                  <option value="none">不切分</option>
                  <option value="four_sentences">凑四句一切</option>
                  <option value="fifty_chars">凑 50 字一切</option>
                  <option value="zh_period">按中文句号切</option>
                  <option value="en_period">按英文句号切</option>
                  <option value="punctuation">按标点符号切</option>
                </select>
                <label htmlFor="local-speed">语速：{localOptions.speed.toFixed(2)}</label>
                <input id="local-speed" type="range" min={0.6} max={1.65} step={0.05} value={localOptions.speed} onChange={(event) => setDraft({ localOptions: { ...localOptions, speed: Number(event.target.value) } })} />
                <label htmlFor="local-pause">句间停顿：{localOptions.pause_seconds.toFixed(2)} 秒</label>
                <input id="local-pause" type="range" min={0.1} max={0.5} step={0.01} value={localOptions.pause_seconds} onChange={(event) => setDraft({ localOptions: { ...localOptions, pause_seconds: Number(event.target.value) } })} />
                <label htmlFor="local-top-k">top_k</label>
                <input id="local-top-k" type="number" min={1} max={100} value={localOptions.top_k} onChange={(event) => setDraft({ localOptions: { ...localOptions, top_k: Number(event.target.value) } })} />
                <label htmlFor="local-top-p">top_p</label>
                <input id="local-top-p" type="number" min={0} max={1} step={0.05} value={localOptions.top_p} onChange={(event) => setDraft({ localOptions: { ...localOptions, top_p: Number(event.target.value) } })} />
                <label htmlFor="local-temperature">temperature</label>
                <input id="local-temperature" type="number" min={0} max={1} step={0.05} value={localOptions.temperature} onChange={(event) => setDraft({ localOptions: { ...localOptions, temperature: Number(event.target.value) } })} />
              </div>
              <div className="field">
                <label>情感控制</label>
                <Dropdown
                  label="情感控制"
                  value={emotionMode}
                  onChange={(value) => setDraft({ emotionMode: value as "auto" | "manual" })}
                  options={[{ value: "auto", label: "自动" }, { value: "manual", label: "手动" }]}
                />
                {emotionMode === "manual" && (
                  <>
                    <label className="sr-only-label">情感标签</label>
                    <Dropdown
                      label="情感标签"
                      value={emotionLabel}
                      onChange={(value) => setDraft({ emotionLabel: value as EmotionLabel })}
                      options={manualOptions.length === 0
                        ? [{ value: emotionLabel, label: `${EMOTION_LABEL_TEXT[emotionLabel] ?? emotionLabel}（无可用参考）` }]
                        : manualOptions.map((value) => ({ value, label: EMOTION_LABEL_TEXT[value] ?? value }))}
                    />
                    {emotionReferenceUnavailable && <p className="hint" role="status">该情绪暂无已审核参考，请改用自动或先添加对应参考。</p>}
                  </>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="field">
                <label htmlFor="cloud-control-mode">情绪控制方式</label>
                <select id="cloud-control-mode" value={cloudControlMode} onChange={(event) => setDraft({ cloudControlMode: event.target.value as typeof cloudControlMode })}>
                  <option value="auto">使用正文作为情绪提示</option>
                  <option value="label">情绪标签</option>
                  <option value="description">自然语言描述</option>
                  <option value="vector">八维情绪向量</option>
                </select>
                {cloudControlMode === "label" && (
                  <>
                    <label htmlFor="cloud-emotion-label">目标情绪</label>
                    <select id="cloud-emotion-label" value={cloudEmotionLabel} onChange={(event) => setDraft({ cloudEmotionLabel: event.target.value as CloudEmotionLabel })}>
                      {CLOUD_EMOTION_LABELS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
                    </select>
                  </>
                )}
                {cloudControlMode === "description" && (
                  <>
                    <label htmlFor="cloud-emotion-description">情绪描述</label>
                    <textarea id="cloud-emotion-description" maxLength={200} value={cloudDescription} onChange={(event) => setDraft({ cloudDescription: event.target.value })} placeholder="例如：压低声音，带着克制而明显的悲伤" />
                    <p className="hint">{cloudDescription.length}/200</p>
                  </>
                )}
                {cloudControlMode === "vector" && (
                  <div className="emotion-vector-editor">
                    <label htmlFor="cloud-vector-mode">向量模式</label>
                    <select id="cloud-vector-mode" value={cloudVectorMode} onChange={(event) => setDraft({ cloudVectorMode: event.target.value as typeof cloudVectorMode })}>
                      <option value="single">单一情绪</option>
                      <option value="mixed">混合情绪</option>
                    </select>
                    {VECTOR_LABELS.map((label, index) => (
                      <label key={label} className="vector-control">{label}：{cloudVector[index].toFixed(2)}
                        <input type="range" min={0} max={1.5} step={0.05} value={cloudVector[index]} onChange={(event) => setDraft({ cloudVector: cloudVector.map((value, vectorIndex) => vectorIndex === index ? Number(event.target.value) : value) })} />
                      </label>
                    ))}
                    <p className="hint">向量总和必须大于 0 且不超过 1.5。</p>
                  </div>
                )}
                <label htmlFor="synthesis-strength">情绪强度：{strength.toFixed(2)}</label>
                <input id="synthesis-strength" type="range" min={0} max={1} step={0.05} value={strength} onChange={(event) => setDraft({ strength: Number(event.target.value) })} />
                <label htmlFor="cloud-sample-rate">输出采样率</label>
                <select id="cloud-sample-rate" value={cloudSampleRate} onChange={(event) => setDraft({ cloudSampleRate: Number(event.target.value) as typeof cloudSampleRate })}>
                  <option value={22050}>22.05 kHz</option>
                  <option value={44100}>44.1 kHz</option>
                  <option value={48000}>48 kHz</option>
                </select>
                <label htmlFor="cloud-speed">语速：{cloudSpeed.toFixed(2)}</label>
                <input id="cloud-speed" type="range" min={0.25} max={4} step={0.05} value={cloudSpeed} onChange={(event) => setDraft({ cloudSpeed: Number(event.target.value) })} />
                <label htmlFor="cloud-gain">音量增益：{cloudGain.toFixed(2)}</label>
                <input id="cloud-gain" aria-label="音量增益" type="range" min={0.1} max={4} step={0.05} value={cloudGain} onChange={(event) => setDraft({ cloudGain: Number(event.target.value) })} />
                <label htmlFor="cloud-interval-silence">句间静音：{cloudIntervalSilence} 毫秒</label>
                <input id="cloud-interval-silence" aria-label="句间静音" type="number" min={0} max={5000} step={10} value={cloudIntervalSilence} onChange={(event) => setDraft({ cloudIntervalSilence: Number(event.target.value) })} />
                <label className="inline-checkbox" htmlFor="cloud-use-random">
                  <input id="cloud-use-random" type="checkbox" checked={cloudUseRandom} onChange={(event) => setDraft({ cloudUseRandom: event.target.checked })} />
                  <span>随机采样</span>
                </label>
              </div>
              <label className="consent-box cloud-consent-box">
                <input type="checkbox" checked={cloudConsent} onChange={(event) => setDraft({ cloudConsent: event.target.checked })} />
                <span>允许将参考音频发送至云端进行情绪合成</span>
              </label>
            </>
          )}
          {plazaVoice && selected === plazaVoice.id && <p className="hint">正在使用广场音色《{plazaVoice.label}》，请确认已获得声音授权后勾选下方确认。</p>}
          <label className="consent-box">
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setDraft({ consent: event.target.checked })}
            />
            <span>我确认已获得声音授权</span>
          </label>
          <button className="primary" type="button" disabled={!canSubmit} onClick={() => void submit()}>
            {submitting ? "提交中…" : "开始安全合成"}
          </button>
          {error && <ErrorPanel error={error} />}
        </section>
        <section className="card synthesis-result-panel" aria-label="合成结果">
          <h2>合成结果</h2>
          <p className="card-lede">只有通过全部安全检查的音频才会在这里发布。</p>
          {jobId ? (
            <>
              <JobProgress jobId={jobId} label="合成任务" />
              {synthesis?.download_ready ? (
                <AudioResultCard
                  src={api.audioUrl(jobId)}
                  downloadName={`合成_${jobId.slice(0, 8)}.wav`}
                  similarity={synthesis.speaker_similarity}
                  warningCodes={synthesis.quality_warning_codes}
                />
              ) : (
                <p className="note" role="status">
                  {synthesis && TERMINAL.has(synthesis.status)
                    ? "该任务未产生可发布的音频，可调整内容后重新提交。"
                    : synthesis?.progress_message ?? "正在生成并完成质量检查，完成后可在此播放或下载。"}
                </p>
              )}
            </>
          ) : (
            <div className="result-placeholder" aria-label="等待创作结果">
              <div className="result-placeholder-ring" aria-hidden="true">
                +
              </div>
              <h3>准备好开始创作</h3>
              <p>完成左侧设置后，合成结果会在这里出现。</p>
              <div className="placeholder-waveform" aria-hidden="true">
                {Array.from({ length: 24 }, (_, index) => (
                  <span key={index} style={{ height: `${22 + ((index * 17) % 52)}%` }} />
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
