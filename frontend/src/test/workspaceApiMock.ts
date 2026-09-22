import { vi } from "vitest";

import type { AuthUser, DatasetPreprocessResponse, VoiceSummary } from "../api/types";

type Item = Record<string, unknown>;

export type WorkspaceMockOptions = {
  authenticated?: boolean;
  currentUser?: Partial<AuthUser>;
  dashboard?: Item;
  health?: Item;
  datasets?: Item[];
  voices?: Array<Item | VoiceSummary>;
  jobs?: Item[];
  syntheses?: Item[];
  adminUsers?: Item[];
  auditEvents?: Item[];
  auditNextBeforeId?: number | null;
  uploadResponse?: Item;
  /** 每次 POST /api/datasets 按序消费；用尽后回退到 uploadResponse/默认。供创建向导辅助参考上传流程使用。 */
  uploadSequence?: Item[];
  preprocessResponse?: DatasetPreprocessResponse | Promise<DatasetPreprocessResponse>;
  segmentsResponse?: Item;
  confirmResult?: Item;
  createdVoice?: Item;
  watermarkVerification?: Item;
  plazaPosts?: Item[];
  plazaComments?: Item[];
  notifications?: Item[];
  unreadCount?: number;
  userProfile?: Item;
  myComments?: Item[];
};

export const LOW_SNR_SEGMENT: Item = {
  snr_db: 11.361,
  snr_warning_db: 20,
  warning_codes: ["SNR_BELOW_RECOMMENDED"],
};

export const ZERO_SHOT_VOICE: VoiceSummary = {
  id: "voice-1",
  display_name: "零样本演示音色",
  dataset_id: "dataset-1",
  status: "ready",
  mode: "zero_shot",
  base_model_id: "gpt-sovits-v2pro-official",
  reference_emotions: ["happy"],
  reference_snr_db: null,
  quality_warning_codes: [],
  can_synthesize: true,
  created_at: "2026-09-12T00:00:00Z",
};

const DEFAULT_PREPROCESS_RESPONSE: DatasetPreprocessResponse = {
  dataset_id: "dataset-1",
  effective_seconds: 7,
  status: "review_required",
  segment_count: 2,
  snr_warning_db: 20,
  warning_codes: ["SNR_BELOW_RECOMMENDED"],
};

const DEFAULT_SEGMENTS: Item = {
  dataset_id: "dataset-1",
  status: "review_required",
  total_segments: 2,
  reviewed_segments: 0,
  effective_seconds: 7,
  reviewed_effective_seconds: 0,
  ready_for_profile: false,
  items: [
    {
      segment_id: "seg_0001",
      order_index: 0,
      relative_path: "datasets/dataset-1/segments/seg_0001.wav",
      duration_seconds: 3.5,
      language: "zh",
      split: "reference",
      auto_transcript: "自动转写文本一。",
      manual_transcript: null,
      effective_transcript: "自动转写文本一。",
      transcript_source: "auto",
      auto_emotion_label: "happy",
      auto_emotion_confidence: 0.9,
      manual_emotion_label: null,
      effective_emotion_label: "happy",
      emotion_source: "auto",
      review_reason: null,
      reviewed: false,
      ...LOW_SNR_SEGMENT,
    },
    {
      segment_id: "seg_0002",
      order_index: 1,
      relative_path: "datasets/dataset-1/segments/seg_0002.wav",
      duration_seconds: 3.5,
      language: "zh",
      split: "reference",
      auto_transcript: "自动转写文本二。",
      manual_transcript: null,
      effective_transcript: "自动转写文本二。",
      transcript_source: "auto",
      auto_emotion_label: "sad",
      auto_emotion_confidence: 0.62,
      manual_emotion_label: null,
      effective_emotion_label: "sad",
      emotion_source: "auto",
      review_reason: null,
      reviewed: false,
    },
  ],
};

function confirmedSegments(base: Item): Item {
  const items = (base.items as Item[]).map((item) => ({ ...item, reviewed: true }));
  return {
    ...base,
    status: "ready_for_profile",
    reviewed_segments: items.length,
    reviewed_effective_seconds: base.effective_seconds,
    ready_for_profile: true,
    items,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const PLAZA_POST_PLACEHOLDER: Item = {
  id: "post-1",
  voice: {
    voice_profile_id: "voice-1",
    display_name: "零样本演示音色",
    status: "ready",
    reference_count: 2,
    reference_emotions: ["happy"],
  },
  author: { user_id: "user-1", display_name: "alice", has_avatar: false },
  description: "简介",
  like_count: 1,
  comment_count: 2,
  favorite_count: 3,
  liked_by_me: false,
  favorited_by_me: true,
  created_at: "2026-09-17T00:00:00Z",
};

function safeJsonParse(body: unknown): Item {
  try {
    return JSON.parse(String(body)) as Item;
  } catch {
    return {};
  }
}

/**
 * 在 fetch 层安装工作台 API 的统一 mock，供组件测试使用。
 * 返回结构遵循零样本客户端公开契约；测试不得各自伪造不一致的返回。
 * 返回被 stub 的 fetch（vi.fn），供断言请求调用序列使用。
 * 注意：客户端训练接口已禁用，mock 不提供 /train 路由——前端任何训练调用都会
 * 以 unexpected request 失败。
 */
export function installWorkspaceApiMock(options: WorkspaceMockOptions = {}) {
  let authenticated = options.authenticated ?? true;
  const confirmedDatasets = new Set<string>();
  let uploadCounter = 0;
  const currentUser: AuthUser = {
    id: "user-1",
    username: "alice",
    role: "user",
    status: "active",
    ...options.currentUser,
  };
  const dashboard = {
    counts: { datasets: 0, voices: 0, jobs: 0, syntheses: 0 },
    recent_jobs: [],
    voices: [],
    readiness: { status: "not_ready", checks: {} },
    ...options.dashboard,
  };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input)).pathname;
    const method = init?.method ?? "GET";
    if (path === "/api/health/ready") return jsonResponse(options.health ?? { status: "not_ready", checks: {} });
    if (path === "/api/auth/me") {
      return authenticated
        ? jsonResponse(currentUser)
        : jsonResponse({ error: { code: "AUTH_REQUIRED", message: "需要登录", details: {}, request_id: "" } }, 401);
    }
    if (path === "/api/auth/register") return jsonResponse(currentUser, 201);
    if (path === "/api/auth/login") { authenticated = true; return jsonResponse(currentUser); }
    if (path === "/api/auth/logout") { authenticated = false; return new Response(null, { status: 204 }); }
    if (path === "/api/dashboard") return jsonResponse(dashboard);
    if (path === "/api/datasets" && method === "GET") return jsonResponse({ items: options.datasets ?? [] });
    if (path === "/api/datasets" && method === "POST") {
      const seq = options.uploadSequence;
      let body: Item;
      if (seq && seq.length > 0) {
        body = seq[Math.min(uploadCounter, seq.length - 1)];
        uploadCounter += 1;
      } else {
        body = options.uploadResponse ?? { dataset_id: "dataset-1", asset_id: "asset-1", content_hash: "hash" };
      }
      return jsonResponse(body, 201);
    }
    if (path.startsWith("/api/datasets/") && path.endsWith("/preprocess")) {
      const dsId = path.split("/")[3];
      const fallback = dsId === "dataset-1"
        ? DEFAULT_PREPROCESS_RESPONSE
        : { ...DEFAULT_PREPROCESS_RESPONSE, dataset_id: dsId };
      const body = options.preprocessResponse ?? fallback;
      return jsonResponse(body instanceof Promise ? await body : body);
    }
    {
      const m = path.match(/^\/api\/datasets\/([^/]+)\/segments$/);
      if (m && method === "GET") {
        const dsId = m[1];
        const base = dsId === "dataset-1"
          ? (options.segmentsResponse ?? DEFAULT_SEGMENTS)
          : { ...DEFAULT_SEGMENTS, dataset_id: dsId };
        return jsonResponse(confirmedDatasets.has(dsId) ? confirmedSegments(base) : base);
      }
    }
    {
      const m = path.match(/^\/api\/datasets\/([^/]+)\/segments\/([^/]+)$/);
      if (m && method === "PATCH") {
        const dsId = m[1];
        const segId = m[2];
        const base = dsId === "dataset-1"
          ? (options.segmentsResponse ?? DEFAULT_SEGMENTS)
          : { ...DEFAULT_SEGMENTS, dataset_id: dsId };
        const items = base.items as Item[];
        const seg = items.find((it) => it.segment_id === segId) ?? items[0];
        const body = JSON.parse(String(init?.body ?? "{}")) as { transcript?: string; emotion_label?: string; language?: string };
        return jsonResponse({
          ...seg,
          manual_transcript: body.transcript ?? null,
          effective_transcript: body.transcript ?? seg.effective_transcript,
          transcript_source: body.transcript ? "manual" : "auto",
          manual_emotion_label: body.emotion_label ?? null,
          effective_emotion_label: body.emotion_label ?? seg.effective_emotion_label,
          emotion_source: body.emotion_label ? "manual" : "auto",
          language: body.language ?? seg.language,
          reviewed: true,
        });
      }
    }
    {
      const m = path.match(/^\/api\/datasets\/([^/]+)\/segments\/confirm$/);
      if (m && method === "POST") {
        const dsId = m[1];
        confirmedDatasets.add(dsId);
        return jsonResponse(
          options.confirmResult ?? { dataset_id: dsId, status: "ready_for_profile", reviewed_effective_seconds: 7 },
        );
      }
    }
    if (path === "/api/voices" && method === "GET") return jsonResponse({ items: options.voices ?? [] });
    if (path === "/api/voices" && method === "POST") return jsonResponse(options.createdVoice ?? ZERO_SHOT_VOICE, 201);
    if (/^\/api\/voices\/[^/]+\/references$/.test(path) && method === "POST") {
      const base = (options.createdVoice as Item | undefined) ?? ZERO_SHOT_VOICE;
      const emotions = (base.reference_emotions as string[] | undefined) ?? [];
      return jsonResponse({ ...base, reference_emotions: [...emotions, "happy"] }, 201);
    }
    if (path === "/api/jobs") return jsonResponse({ items: options.jobs ?? [] });
    if (path.startsWith("/api/jobs/")) return jsonResponse({ id: "job-1", kind: "synthesize", status: "queued", error_code: null, public_message: null });
    if (path === "/api/syntheses" && method === "GET") return jsonResponse({ items: options.syntheses ?? [] });
    if (path === "/api/syntheses" && method === "POST") return jsonResponse({ job_id: "synthesis-1", status: "queued" }, 202);
    if (path === "/api/safety/detect-watermark") return jsonResponse(options.watermarkVerification ?? { job_id: "synthesis-1", probability: 0.98, payload: 1, payload_matches_job: true });
    if (path === "/api/admin/users" && method === "GET") return jsonResponse({ items: options.adminUsers ?? [] });
    if (path.startsWith("/api/admin/users/") && method === "PATCH") return jsonResponse(options.adminUsers?.[0] ?? { id: "user-2", username: "alice", role: "user", status: "disabled" });
    if (path === "/api/admin/audit-events") return jsonResponse({ items: options.auditEvents ?? [], next_before_id: options.auditNextBeforeId ?? null });
    if (path === "/api/plaza/posts" && method === "GET") {
      const url = new URL(String(input));
      let items = [...(options.plazaPosts ?? [])];
      if (url.searchParams.get("mine") === "true")
        items = items.filter((item) => (item.author as Item).user_id === currentUser.id);
      if (url.searchParams.get("favorited") === "true")
        items = items.filter((item) => Boolean(item.favorited_by_me));
      const q = url.searchParams.get("q");
      if (q)
        items = items.filter(
          (item) =>
            String((item.voice as Item).display_name).includes(q) ||
            String(item.description ?? "").includes(q),
        );
      return jsonResponse({ items, total: items.length });
    }
    if (path === "/api/plaza/posts" && method === "POST")
      return jsonResponse(
        { ...PLAZA_POST_PLACEHOLDER, ...safeJsonParse(init?.body) },
        201,
      );
    if (/^\/api\/plaza\/posts\/[^/]+\/comments$/.test(path) && method === "GET")
      return jsonResponse({ items: options.plazaComments ?? [], total: (options.plazaComments ?? []).length });
    if (/^\/api\/plaza\/posts\/[^/]+\/comments$/.test(path) && method === "POST")
      return jsonResponse({ id: "comment-new", post_id: "post-1", author: { user_id: "user-1", display_name: "alice", has_avatar: false }, content: "新评论", created_at: "2026-09-17T00:00:00Z" }, 201);
    if (/^\/api\/plaza\/posts\/[^/]+\/(likes|favorites)$/.test(path))
      return jsonResponse({ ok: true });
    if (/^\/api\/plaza\/posts\/[^/]+$/.test(path) && method === "DELETE")
      return jsonResponse({ deleted: true });
    if (path === "/api/plaza/comments" && method === "GET")
      return jsonResponse({ items: options.myComments ?? [], total: (options.myComments ?? []).length });
    if (path === "/api/notifications" && method === "GET")
      return jsonResponse({ items: options.notifications ?? [], total: (options.notifications ?? []).length });
    if (path === "/api/notifications/unread-count")
      return jsonResponse({ count: options.unreadCount ?? 0 });
    if (path === "/api/notifications/mark-read" && method === "POST")
      return jsonResponse({ updated: 0 });
    if (path === "/api/users/me/profile" && method === "GET")
      return jsonResponse(
        options.userProfile ?? {
          user_id: currentUser.id,
          username: currentUser.username,
          display_name: currentUser.username,
          bio: null,
          has_avatar: false,
          stats: { published: 0, likes_received: 0 },
        },
      );
    if (path === "/api/users/me/profile" && (method === "PATCH" || method === "POST"))
      return jsonResponse(
        options.userProfile ?? {
          user_id: currentUser.id,
          username: currentUser.username,
          display_name: "小雅",
          bio: null,
          has_avatar: false,
          stats: { published: 0, likes_received: 0 },
        },
      );
    throw new Error(`unexpected request: ${method} ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
