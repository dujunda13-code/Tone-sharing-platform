import type {
  AdminUserListResponse,
  AdminUserSummary,
  ApiErrorBody,
  AuditEventListResponse,
  AuthUser,
  DashboardResponse,
  DatasetListResponse,
  DatasetPreprocessResponse,
  SegmentListResponse,
  SegmentReviewPayload,
  SegmentSummary,
  DatasetUploadResponse,
  EmotionControl,
  CloudEmotionOptions,
  HealthResponse,
  JobListResponse,
  JobStatus,
  JobSummary,
  LocalSynthesisOptions,
  SynthesisLanguage,
  SynthesisMode,
  NotificationListResponse,
  PlazaComment,
  PlazaCommentListResponse,
  PlazaPost,
  PlazaPostListResponse,
  SynthesisListResponse,
  SynthesisSummary,
  UnreadCountResponse,
  UserProfileSummary,
  VoiceListResponse,
  VoiceProfile,
  WatermarkResponse,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

function redactImplementationDetails(message: string): string {
  return message
    .replace(/[A-Za-z]:\\[^\s]+/g, "文件位置")
    .replace(/\/(?:Users|home|var|tmp)\/[^\s]+/g, "文件位置")
    .replace(/Traceback[\s\S]*/g, "服务处理失败，请稍后重试");
}

/** 后端 public_message 统一经此渲染：去除实现措辞与位置信息。 */
export function toUserFacingMessage(message: string): string {
  return redactImplementationDetails(message).replace(/本地|本机/g, "系统");
}

export class ApiRequestError extends Error {
  readonly code: string;
  readonly requestId: string;

  constructor(body: ApiErrorBody, status: number) {
    super(redactImplementationDetails(body.error.message));
    this.name = "ApiRequestError";
    this.code = body.error.code || `HTTP_${status}`;
    this.requestId = body.error.request_id;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  const body = (await response.json().catch(() => ({}))) as T | ApiErrorBody;
  if (!response.ok) {
    const errorBody = body as ApiErrorBody;
    throw new ApiRequestError(
      errorBody.error
        ? errorBody
        : { error: { code: `HTTP_${response.status}`, message: "请求失败", details: {}, request_id: "" } },
      response.status,
    );
  }
  return body as T;
}

export const api = {
  health: async () => {
    const response = await fetch(`${API_BASE}/api/health/ready`, {
      credentials: "include",
      headers: { Accept: "application/json" },
    });
    return (await response.json()) as HealthResponse;
  },
  me: () => request<AuthUser>("/api/auth/me"),
  login: (username: string, password: string) =>
    request<AuthUser>("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }),
  register: (username: string, password: string) =>
    request<AuthUser>("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  dashboard: () => request<DashboardResponse>("/api/dashboard"),
  datasets: () => request<DatasetListResponse>("/api/datasets"),
  voices: () => request<VoiceListResponse>("/api/voices"),
  jobs: (status?: JobStatus) =>
    request<JobListResponse>(
      status ? `/api/jobs?status=${encodeURIComponent(status)}` : "/api/jobs",
    ),
  jobSummary: (jobId: string) => request<JobSummary>(`/api/jobs/${encodeURIComponent(jobId)}`),
  synthesisList: () => request<SynthesisListResponse>("/api/syntheses"),
  uploadDataset: (file: File, consentConfirmed: true) => {
    const form = new FormData();
    form.append("file", file);
    form.append("consent_confirmed", String(consentConfirmed));
    return request<DatasetUploadResponse>("/api/datasets", { method: "POST", body: form });
  },
  preprocess: (datasetId: string) =>
    request<DatasetPreprocessResponse>(`/api/datasets/${encodeURIComponent(datasetId)}/preprocess`, {
      method: "POST",
    }),
  segments: (datasetId: string) =>
    request<SegmentListResponse>(`/api/datasets/${encodeURIComponent(datasetId)}/segments`),
  reviewSegment: (datasetId: string, segmentId: string, payload: SegmentReviewPayload) =>
    request<SegmentSummary>(
      `/api/datasets/${encodeURIComponent(datasetId)}/segments/${encodeURIComponent(segmentId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    ),
  confirmSegments: (datasetId: string, reason: string) =>
    request<{ dataset_id: string; status: string; reviewed_effective_seconds: number }>(
      `/api/datasets/${encodeURIComponent(datasetId)}/segments/confirm`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      },
    ),
  createVoice: (datasetId: string, displayName: string, referenceName?: string) =>
    request<VoiceProfile>("/api/voices", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: datasetId,
        display_name: displayName.trim(),
        reference_name: referenceName?.trim() || undefined,
      }),
    }),
  addVoiceReference: (profileId: string, datasetId: string, referenceName?: string) =>
    request<VoiceProfile>(`/api/voices/${encodeURIComponent(profileId)}/references`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: datasetId,
        reference_name: referenceName?.trim() || undefined,
      }),
    }),
  deleteVoiceReference: (profileId: string, referenceId: string) =>
    request<void>(
      `/api/voices/${encodeURIComponent(profileId)}/references/${encodeURIComponent(referenceId)}`,
      { method: "DELETE" }
    ),
  synthesize: (body: {
    voice_profile_id: string;
    text: string;
    text_lang: SynthesisLanguage;
    emotion: EmotionControl;
    consent_confirmed: true;
    synthesis_mode: SynthesisMode;
    local_options: LocalSynthesisOptions;
    cloud_options?: CloudEmotionOptions;
    cloud_processing_confirmed: boolean;
  }) =>
    request<{ job_id: string; status: string }>("/api/syntheses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  evaluate: (profileId: string) =>
    request<{ job_id: string; status: string }>(`/api/voices/${encodeURIComponent(profileId)}/evaluate`, {
      method: "POST",
    }),
  job: (jobId: string) => request<JobSummary>(`/api/jobs/${encodeURIComponent(jobId)}`),
  synthesisJob: (jobId: string) =>
    request<SynthesisSummary>(`/api/syntheses/${encodeURIComponent(jobId)}`),
  detectWatermark: (jobId: string) =>
    request<WatermarkResponse>("/api/safety/detect-watermark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId }),
    }),
  audioUrl: (jobId: string) => `${API_BASE}/api/syntheses/${encodeURIComponent(jobId)}/audio`,
  adminUsers: () => request<AdminUserListResponse>("/api/admin/users"),
  updateAdminUserStatus: (userId: string, status: "active" | "disabled") =>
    request<AdminUserSummary>(`/api/admin/users/${encodeURIComponent(userId)}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
  adminAuditEvents: (limit = 20, beforeId?: number) => {
    const cursor = beforeId === undefined ? "" : `&before_id=${encodeURIComponent(beforeId)}`;
    return request<AuditEventListResponse>(`/api/admin/audit-events?limit=${limit}${cursor}`);
  },
  plazaPosts: (params?: { mine?: boolean; favorited?: boolean; q?: string; sort?: "newest" | "likes" }) => {
    const search = new URLSearchParams();
    if (params?.mine) search.set("mine", "true");
    if (params?.favorited) search.set("favorited", "true");
    if (params?.q) search.set("q", params.q);
    if (params?.sort) search.set("sort", params.sort);
    const query = search.toString();
    return request<PlazaPostListResponse>(`/api/plaza/posts${query ? `?${query}` : ""}`);
  },
  plazaPublish: (body: { voice_profile_id: string; description?: string }) =>
    request<PlazaPost>("/api/plaza/posts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  plazaDelete: (postId: string) =>
    request<{ deleted: boolean }>(`/api/plaza/posts/${encodeURIComponent(postId)}`, { method: "DELETE" }),
  plazaComments: (postId: string) =>
    request<PlazaCommentListResponse>(`/api/plaza/posts/${encodeURIComponent(postId)}/comments`),
  plazaAddComment: (postId: string, content: string) =>
    request<PlazaComment>(`/api/plaza/posts/${encodeURIComponent(postId)}/comments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    }),
  plazaLike: (postId: string) =>
    request<{ ok: boolean }>(`/api/plaza/posts/${encodeURIComponent(postId)}/likes`, { method: "POST" }),
  plazaUnlike: (postId: string) =>
    request<{ ok: boolean }>(`/api/plaza/posts/${encodeURIComponent(postId)}/likes`, { method: "DELETE" }),
  plazaFavorite: (postId: string) =>
    request<{ ok: boolean }>(`/api/plaza/posts/${encodeURIComponent(postId)}/favorites`, { method: "POST" }),
  plazaUnfavorite: (postId: string) =>
    request<{ ok: boolean }>(`/api/plaza/posts/${encodeURIComponent(postId)}/favorites`, { method: "DELETE" }),
  plazaDownloadUrl: (postId: string) =>
    `${API_BASE}/api/plaza/posts/${encodeURIComponent(postId)}/download`,
  plazaReferenceDownloadUrl: (postId: string) =>
    `${API_BASE}/api/plaza/posts/${encodeURIComponent(postId)}/download/reference`,
  plazaPreviewUrl: (postId: string) =>
    `${API_BASE}/api/plaza/posts/${encodeURIComponent(postId)}/preview`,
  plazaImport: (postId: string, authorizationConfirmed: true) =>
    request<{ voice_profile_id: string; display_name: string }>(
      `/api/plaza/posts/${encodeURIComponent(postId)}/import`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ authorization_confirmed: authorizationConfirmed }),
      },
    ),
  myPlazaComments: () => request<PlazaCommentListResponse>("/api/plaza/comments"),
  notifications: (params?: { limit?: number; offset?: number }) => {
    const search = new URLSearchParams();
    if (params?.limit) search.set("limit", String(params.limit));
    if (params?.offset) search.set("offset", String(params.offset));
    const query = search.toString();
    return request<NotificationListResponse>(`/api/notifications${query ? `?${query}` : ""}`);
  },
  notificationsUnreadCount: () => request<UnreadCountResponse>("/api/notifications/unread-count"),
  notificationsMarkRead: (body: { notification_ids?: string[]; all?: boolean }) =>
    request<{ updated: number }>("/api/notifications/mark-read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  userProfile: () => request<UserProfileSummary>("/api/users/me/profile"),
  updateUserProfile: (body: { display_name: string; bio?: string | null }) =>
    request<UserProfileSummary>("/api/users/me/profile", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  uploadAvatar: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UserProfileSummary>("/api/users/me/avatar", { method: "POST", body: form });
  },
  avatarUrl: (userId: string) => `${API_BASE}/api/avatars/${encodeURIComponent(userId)}`,
};

export function getApiError(error: unknown): { code: string; message: string; requestId: string } {
  if (error instanceof ApiRequestError) {
    return { code: error.code, message: error.message, requestId: error.requestId };
  }
  return { code: "SERVICE_UNAVAILABLE", message: "服务暂时不可用，请稍后重试。", requestId: "" };
}
