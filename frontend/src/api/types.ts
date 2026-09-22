import type { components } from "./generated";

export type JobStatus = components["schemas"]["JobStatus"];
export type JobKind = components["schemas"]["JobKind"];
export type Language = "zh" | "en";
export type SynthesisLanguage = "zh" | "en" | "ja" | "es" | "ar";
export type SynthesisMode = "local" | "emotion_api";

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details: unknown;
    request_id: string;
  };
}

export interface DatasetUploadResponse {
  dataset_id: string;
  asset_id: string;
  content_hash: string;
  effective_seconds?: number;
  rejected_segments?: Array<{ id: string; reason: string }>;
  emotion_labels?: Array<{ segment_id: string; source: "auto" | "manual"; label: string }>;
}

export type VoiceProfile = components["schemas"]["VoiceProfileResponse"];
export type WatermarkResponse = components["schemas"]["WatermarkVerificationResponse"];
export type HealthResponse = components["schemas"]["ReadyResponse"];

export interface AuthUser {
  id: string;
  username: string;
  role: "user" | "admin";
  status: string;
}

/**
 * 以下工作台摘要类型对应《本地音色创作工作台设计》§5 的公开 API 契约。
 * 后端 owner 过滤列表/概览/管理员接口尚未实现（已记录为交接日志中的 API 缺口）；
 * 后端落地后应从 OpenAPI 重新生成并核对这些字段，前端不得自行放宽脱敏边界。
 */

export interface DatasetPreprocessResponse {
  dataset_id: string;
  effective_seconds: number;
  status: string;
  segment_count: number;
  snr_warning_db: number;
  warning_codes: string[];
}

export interface DatasetSummary {
  id: string;
  authorization_confirmed: boolean;
  status: string;
  effective_seconds: number | null;
  asset_count: number;
  created_at: string;
}

export interface SegmentSummary {
  segment_id: string;
  order_index: number;
  relative_path: string;
  duration_seconds: number;
  language: string;
  split: string;
  auto_transcript: string;
  manual_transcript: string | null;
  effective_transcript: string;
  transcript_source: string;
  auto_emotion_label: string | null;
  auto_emotion_confidence: number | null;
  manual_emotion_label: string | null;
  effective_emotion_label: string | null;
  emotion_source: string;
  review_reason: string | null;
  reviewed: boolean;
  snr_db: number | null;
  snr_warning_db: number;
  warning_codes: string[];
}

export interface SegmentListResponse {
  dataset_id: string;
  status: string;
  total_segments: number;
  reviewed_segments: number;
  effective_seconds: number | null;
  reviewed_effective_seconds: number;
  ready_for_profile: boolean;
  items: SegmentSummary[];
}

export interface SegmentReviewPayload {
  transcript?: string;
  emotion_label?: string;
  language?: Language;
  reason: string;
}

export type VoiceReferenceItem = components["schemas"]["VoiceReferenceItemResponse"];

export interface VoiceSummary {
  id: string;
  display_name: string;
  dataset_id: string;
  status: string;
  created_at: string;
  can_synthesize: boolean;
  mode: "zero_shot" | null;
  base_model_id: string | null;
  reference_emotions: string[];
  reference_snr_db: number | null;
  quality_warning_codes: string[];
  references?: VoiceReferenceItem[];
}

export interface JobSummary {
  id: string;
  kind: JobKind;
  status: JobStatus;
  error_code: string | null;
  public_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface SynthesisSummary {
  job_id: string;
  voice_profile_id: string | null;
  text_lang: SynthesisLanguage | null;
  status: JobStatus;
  download_ready: boolean;
  watermark_probability: number | null;
  fingerprint_anomaly: boolean | null;
  speaker_similarity: number | null;
  quality_warning_codes: string[];
  progress_message: string | null;
  created_at: string;
}

export interface ReadinessCheckSummary {
  ok: boolean;
  message: string;
}

export interface ReadinessSummary {
  status: "ready" | "not_ready";
  checks: Record<string, ReadinessCheckSummary>;
}

export interface DashboardCounts {
  datasets: number;
  voices: number;
  jobs: number;
  active_jobs: number;
  syntheses: number;
}

export interface DashboardResponse {
  counts: DashboardCounts;
  recent_jobs: JobSummary[];
  voices: VoiceSummary[];
  readiness: ReadinessSummary;
}

export interface DatasetListResponse {
  items: DatasetSummary[];
}

export interface VoiceListResponse {
  items: VoiceSummary[];
}

export interface JobListResponse {
  items: JobSummary[];
}

export interface SynthesisListResponse {
  items: SynthesisSummary[];
}

export interface AdminUserSummary {
  id: string;
  username: string;
  role: "user" | "admin";
  status: "active" | "disabled";
  created_at: string;
  dataset_count: number;
  voice_count: number;
  job_count: number;
}

export interface AuditEventSummary {
  id: number;
  event_type: string;
  subject_id: string | null;
  metadata: Record<string, string>;
  created_at: string;
}

export interface AdminUserListResponse {
  items: AdminUserSummary[];
}

export interface AuditEventListResponse {
  items: AuditEventSummary[];
  next_before_id: number | null;
}

export type EmotionLabel =
  | "neutral"
  | "happy"
  | "sad"
  | "angry"
  | "fearful"
  | "disgusted"
  | "surprised"
  | "other";

export interface EmotionControl {
  mode: "auto" | "manual";
  strength: number;
  label?: EmotionLabel;
}

export type CloudEmotionLabel =
  | "neutral"
  | "happy"
  | "angry"
  | "sad"
  | "afraid"
  | "disgusted"
  | "melancholic"
  | "surprised"
  | "calm";

export interface LocalSynthesisOptions {
  cut_method: "none" | "four_sentences" | "fifty_chars" | "zh_period" | "en_period" | "punctuation";
  speed: number;
  pause_seconds: number;
  top_k: number;
  top_p: number;
  temperature: number;
}

export interface CloudEmotionOptions {
  control_mode: "auto" | "label" | "description" | "vector";
  label?: CloudEmotionLabel;
  description?: string;
  emotion_strength: number;
  sample_rate: 22050 | 44100 | 48000;
  speed: number;
  gain: number;
  use_random: boolean;
  interval_silence: number;
  emotion_vector?: number[];
  emotion_vector_mode?: "single" | "mixed";
}

export type PlazaPost = components["schemas"]["PlazaPostSummary"];
export type PlazaPostListResponse = components["schemas"]["PlazaPostListResponse"];
export type PlazaComment = components["schemas"]["PlazaCommentSummary"];
export type PlazaCommentListResponse = components["schemas"]["PlazaCommentListResponse"];
export type NotificationItem = components["schemas"]["NotificationSummary"];
export type NotificationListResponse = components["schemas"]["NotificationListResponse"];
export type UnreadCountResponse = components["schemas"]["UnreadCountResponse"];
export type UserProfileSummary = components["schemas"]["UserProfileResponse"];
