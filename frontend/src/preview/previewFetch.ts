type PreviewRecord = Record<string, unknown>;

export type PreviewFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

const PREVIEW_USER = {
  id: "preview-user",
  username: "preview",
  role: "admin" as const,
  status: "active",
};

const PREVIEW_VOICE = {
  id: "preview-voice",
  display_name: "零样本演示音色",
  dataset_id: "preview-dataset",
  status: "ready",
  mode: "zero_shot" as const,
  base_model_id: "gpt-sovits-20250606v2pro",
  reference_emotions: ["happy", "neutral"],
  reference_snr_db: 24.6,
  quality_warning_codes: [],
  can_synthesize: true,
  created_at: "2026-09-12T00:00:00Z",
  references: [],
};

const PREVIEW_DATASET = {
  id: "preview-dataset",
  authorization_confirmed: true,
  status: "ready_for_profile",
  effective_seconds: 7,
  asset_count: 1,
  created_at: "2026-09-12T00:00:00Z",
};

const PREVIEW_SEGMENTS = [
  {
    segment_id: "preview-segment-1",
    order_index: 0,
    relative_path: "datasets/preview-dataset/segments/segment-1.wav",
    duration_seconds: 3.5,
    language: "zh",
    split: "reference",
    auto_transcript: "这是一段用于界面预览的参考语音。",
    manual_transcript: null,
    effective_transcript: "这是一段用于界面预览的参考语音。",
    transcript_source: "auto",
    auto_emotion_label: "happy",
    auto_emotion_confidence: 0.94,
    manual_emotion_label: null,
    effective_emotion_label: "happy",
    emotion_source: "auto",
    review_reason: null,
    reviewed: true,
    snr_db: 24.6,
    snr_warning_db: 20,
    warning_codes: [],
  },
  {
    segment_id: "preview-segment-2",
    order_index: 1,
    relative_path: "datasets/preview-dataset/segments/segment-2.wav",
    duration_seconds: 3.5,
    language: "zh",
    split: "reference",
    auto_transcript: "预览模式不会上传或处理真实音频。",
    manual_transcript: null,
    effective_transcript: "预览模式不会上传或处理真实音频。",
    transcript_source: "auto",
    auto_emotion_label: "neutral",
    auto_emotion_confidence: 0.91,
    manual_emotion_label: null,
    effective_emotion_label: "neutral",
    emotion_source: "auto",
    review_reason: null,
    reviewed: true,
    snr_db: 25.1,
    snr_warning_db: 20,
    warning_codes: [],
  },
];

const PREVIEW_JOB = {
  id: "preview-job",
  kind: "synthesize",
  status: "succeeded",
  error_code: null,
  public_message: "预览合成已完成",
  created_at: "2026-09-12T00:02:00Z",
  started_at: "2026-09-12T00:02:01Z",
  finished_at: "2026-09-12T00:02:03Z",
};

const PREVIEW_SYNTHESIS = {
  job_id: "preview-job",
  voice_profile_id: PREVIEW_VOICE.id,
  text_lang: "zh",
  status: "succeeded",
  download_ready: true,
  watermark_probability: 0.98,
  fingerprint_anomaly: false,
  speaker_similarity: 0.94,
  quality_warning_codes: [],
  progress_message: "已完成来源水印与声音指纹复检",
  created_at: "2026-09-12T00:02:00Z",
};

const PREVIEW_POST = {
  id: "preview-post",
  voice: {
    voice_profile_id: PREVIEW_VOICE.id,
    display_name: PREVIEW_VOICE.display_name,
    status: "ready",
    reference_count: 2,
    reference_emotions: PREVIEW_VOICE.reference_emotions,
  },
  author: { user_id: PREVIEW_USER.id, display_name: "预览用户", has_avatar: false },
  description: "用于查看音色广场布局的演示音色。",
  like_count: 12,
  comment_count: 1,
  favorite_count: 4,
  liked_by_me: false,
  favorited_by_me: false,
  created_at: "2026-09-12T00:01:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function emptyAudioResponse(): Response {
  return new Response(new Blob([], { type: "audio/wav" }), {
    status: 200,
    headers: { "Content-Type": "audio/wav" },
  });
}

function parseJson(body: unknown): PreviewRecord {
  if (typeof body !== "string") return {};
  try {
    const value = JSON.parse(body) as unknown;
    return value && typeof value === "object" ? value as PreviewRecord : {};
  } catch {
    return {};
  }
}

function pathOf(input: RequestInfo | URL): { path: string; url: URL } {
  const url = new URL(String(input), "http://127.0.0.1:5173");
  return { path: url.pathname, url };
}

/**
 * 返回只在独立前端副本中使用的本地 API 模拟器。
 * 它只响应 /api/*，不启动服务、不读写数据库、不上传文件，也不调用真实后端。
 */
export function createPreviewFetch(fallback?: typeof fetch): PreviewFetch {
  let authenticated = true;
  let datasetCounter = 0;
  let synthesisCounter = 0;
  let datasets: PreviewRecord[] = [PREVIEW_DATASET];
  let voices: PreviewRecord[] = [PREVIEW_VOICE];
  let jobs: PreviewRecord[] = [PREVIEW_JOB];
  let syntheses: PreviewRecord[] = [PREVIEW_SYNTHESIS];
  let posts: PreviewRecord[] = [PREVIEW_POST];

  return async (input, init) => {
    const { path, url } = pathOf(input);
    const method = (init?.method ?? "GET").toUpperCase();

    if (!path.startsWith("/api/")) {
      if (fallback) return fallback(input, init);
      return jsonResponse({ error: { code: "PREVIEW_ROUTE_NOT_FOUND", message: "预览资源不存在" } }, 404);
    }

    if (path === "/api/health/ready") {
      return jsonResponse({
        status: "ready",
        checks: {
          database: { ok: true, message: "预览数据" },
          storage: { ok: true, message: "预览数据" },
          gpu: { ok: true, message: "预览数据" },
          gpt_sovits: { ok: true, message: "预览数据" },
          emotion2vec: { ok: true, message: "预览数据" },
          audioseal: { ok: true, message: "预览数据" },
        },
      });
    }

    if (path === "/api/auth/me") {
      return authenticated
        ? jsonResponse(PREVIEW_USER)
        : jsonResponse({ error: { code: "AUTH_REQUIRED", message: "需要登录", details: {}, request_id: "" } }, 401);
    }
    if (path === "/api/auth/login" || path === "/api/auth/register") {
      authenticated = true;
      return jsonResponse(PREVIEW_USER, path.endsWith("register") ? 201 : 200);
    }
    if (path === "/api/auth/logout") {
      authenticated = false;
      return new Response(null, { status: 204 });
    }

    if (path === "/api/dashboard") {
      return jsonResponse({
        counts: {
          datasets: datasets.length,
          voices: voices.length,
          jobs: jobs.length,
          active_jobs: 0,
          syntheses: syntheses.length,
        },
        recent_jobs: jobs,
        voices,
        readiness: { status: "ready", checks: { models: { ok: true, message: "预览数据" } } },
      });
    }

    if (path === "/api/datasets" && method === "GET") return jsonResponse({ items: datasets });
    if (path === "/api/datasets" && method === "POST") {
      datasetCounter += 1;
      const id = `preview-dataset-${datasetCounter}`;
      const dataset = { ...PREVIEW_DATASET, id, status: "uploaded", asset_count: 1 };
      datasets = [dataset, ...datasets];
      return jsonResponse({ dataset_id: id, asset_id: `preview-asset-${datasetCounter}`, content_hash: "preview-hash", effective_seconds: 7 }, 201);
    }
    {
      const match = path.match(/^\/api\/datasets\/([^/]+)\/preprocess$/);
      if (match && method === "POST") {
        return jsonResponse({ dataset_id: match[1], effective_seconds: 7, status: "review_required", segment_count: 2, snr_warning_db: 20, warning_codes: [] });
      }
    }
    {
      const match = path.match(/^\/api\/datasets\/([^/]+)\/segments$/);
      if (match && method === "GET") {
        return jsonResponse({ dataset_id: match[1], status: "ready_for_profile", total_segments: 2, reviewed_segments: 2, effective_seconds: 7, reviewed_effective_seconds: 7, ready_for_profile: true, items: PREVIEW_SEGMENTS });
      }
    }
    {
      const match = path.match(/^\/api\/datasets\/([^/]+)\/segments\/([^/]+)$/);
      if (match && method === "PATCH") {
        const body = parseJson(init?.body);
        const segment = PREVIEW_SEGMENTS.find((item) => item.segment_id === match[2]) ?? PREVIEW_SEGMENTS[0];
        return jsonResponse({
          ...segment,
          manual_transcript: body.transcript ?? null,
          effective_transcript: body.transcript ?? segment.effective_transcript,
          transcript_source: body.transcript ? "manual" : "auto",
          manual_emotion_label: body.emotion_label ?? null,
          effective_emotion_label: body.emotion_label ?? segment.effective_emotion_label,
          emotion_source: body.emotion_label ? "manual" : "auto",
          language: body.language ?? segment.language,
          reviewed: true,
        });
      }
    }
    if (path.match(/^\/api\/datasets\/[^/]+\/segments\/confirm$/) && method === "POST") {
      return jsonResponse({ dataset_id: path.split("/")[3], status: "ready_for_profile", reviewed_effective_seconds: 7 });
    }

    if (path === "/api/voices" && method === "GET") return jsonResponse({ items: voices });
    if (path === "/api/voices" && method === "POST") {
      const body = parseJson(init?.body);
      const voice = {
        ...PREVIEW_VOICE,
        id: `preview-voice-${voices.length + 1}`,
        dataset_id: String(body.dataset_id ?? "preview-dataset"),
        display_name: String(body.display_name ?? "预览音色"),
      };
      voices = [voice, ...voices];
      return jsonResponse(voice, 201);
    }
    if (path.match(/^\/api\/voices\/[^/]+\/references$/) && method === "POST") return jsonResponse(PREVIEW_VOICE, 201);
    if (path.match(/^\/api\/voices\/[^/]+\/references\/[^/]+$/) && method === "DELETE") return new Response(null, { status: 204 });

    if (path === "/api/jobs" && method === "GET") {
      const status = url.searchParams.get("status");
      return jsonResponse({ items: status ? jobs.filter((job) => job.status === status) : jobs });
    }
    if (path.match(/^\/api\/jobs\/[^/]+$/) && method === "GET") return jsonResponse(jobs[0]);
    if (path === "/api/syntheses" && method === "GET") return jsonResponse({ items: syntheses });
    if (path === "/api/syntheses" && method === "POST") {
      synthesisCounter += 1;
      const id = `preview-job-${synthesisCounter}`;
      const body = parseJson(init?.body);
      const job = { ...PREVIEW_JOB, id, created_at: new Date().toISOString(), finished_at: new Date().toISOString() };
      const synthesis = { ...PREVIEW_SYNTHESIS, job_id: id, voice_profile_id: String(body.voice_profile_id ?? PREVIEW_VOICE.id), created_at: job.created_at };
      jobs = [job, ...jobs];
      syntheses = [synthesis, ...syntheses];
      return jsonResponse({ job_id: id, status: "succeeded" }, 202);
    }
    if (path.match(/^\/api\/syntheses\/[^/]+$/) && method === "GET") {
      const id = path.split("/")[3];
      return jsonResponse(syntheses.find((item) => item.job_id === id) ?? PREVIEW_SYNTHESIS);
    }
    if (path.match(/^\/api\/syntheses\/[^/]+\/audio$/)) return emptyAudioResponse();
    if (path === "/api/safety/detect-watermark" && method === "POST") {
      const body = parseJson(init?.body);
      return jsonResponse({ job_id: String(body.job_id ?? PREVIEW_JOB.id), probability: 0.98, payload: 1, payload_matches_job: true });
    }

    if (path === "/api/admin/users" && method === "GET") {
      return jsonResponse({ items: [
        { id: PREVIEW_USER.id, username: PREVIEW_USER.username, role: "admin", status: "active", created_at: "2026-09-01T00:00:00Z", dataset_count: datasets.length, voice_count: voices.length, job_count: jobs.length },
        { id: "preview-member", username: "member", role: "user", status: "active", created_at: "2026-09-02T00:00:00Z", dataset_count: 1, voice_count: 1, job_count: 0 },
      ] });
    }
    if (path.match(/^\/api\/admin\/users\/[^/]+\/status$/) && method === "PATCH") return jsonResponse({ id: "preview-member", username: "member", role: "user", status: "disabled", created_at: "2026-09-02T00:00:00Z", dataset_count: 1, voice_count: 1, job_count: 0 });
    if (path === "/api/admin/audit-events") return jsonResponse({ items: [{ id: 1, event_type: "admin.user_status_changed", subject_id: "preview-member", metadata: { status: "active" }, created_at: "2026-09-12T00:00:00Z" }], next_before_id: null });

    if (path === "/api/plaza/posts" && method === "GET") {
      let items = [...posts];
      if (url.searchParams.get("mine") === "true") items = items.filter((item) => (item.author as PreviewRecord).user_id === PREVIEW_USER.id);
      if (url.searchParams.get("favorited") === "true") items = items.filter((item) => Boolean(item.favorited_by_me));
      const query = url.searchParams.get("q");
      if (query) items = items.filter((item) => String((item.voice as PreviewRecord).display_name).includes(query) || String(item.description ?? "").includes(query));
      return jsonResponse({ items, total: items.length });
    }
    if (path === "/api/plaza/posts" && method === "POST") return jsonResponse({ ...PREVIEW_POST, description: parseJson(init?.body).description ?? "预览发布" }, 201);
    if (path.match(/^\/api\/plaza\/posts\/[^/]+\/comments$/) && method === "GET") return jsonResponse({ items: [{ id: "preview-comment", post_id: "preview-post", author: { user_id: "preview-member", display_name: "体验用户", has_avatar: false }, content: "这个音色很有辨识度。", voice_name: PREVIEW_VOICE.display_name, created_at: "2026-09-12T00:01:30Z" }], total: 1 });
    if (path.match(/^\/api\/plaza\/posts\/[^/]+\/comments$/) && method === "POST") return jsonResponse({ id: "preview-comment-new", post_id: "preview-post", author: { user_id: PREVIEW_USER.id, display_name: "预览用户", has_avatar: false }, content: "预览评论", created_at: new Date().toISOString() }, 201);
    if (path.match(/^\/api\/plaza\/posts\/[^/]+\/(likes|favorites)$/)) return jsonResponse({ ok: true });
    if (path.match(/^\/api\/plaza\/posts\/[^/]+$/) && method === "DELETE") return jsonResponse({ deleted: true });
    if (path === "/api/plaza/comments" && method === "GET") return jsonResponse({ items: [], total: 0 });
    if (path.match(/^\/api\/plaza\/posts\/[^/]+\/preview$/)) return emptyAudioResponse();
    if (path.match(/^\/api\/plaza\/posts\/[^/]+\/import$/) && method === "POST") return jsonResponse({ voice_profile_id: PREVIEW_VOICE.id, display_name: PREVIEW_VOICE.display_name });

    if (path === "/api/notifications" && method === "GET") return jsonResponse({ items: [], total: 0 });
    if (path === "/api/notifications/unread-count") return jsonResponse({ count: 0 });
    if (path === "/api/notifications/mark-read" && method === "POST") return jsonResponse({ updated: 0 });
    if (path === "/api/users/me/profile" && method === "GET") return jsonResponse({ user_id: PREVIEW_USER.id, username: PREVIEW_USER.username, display_name: "预览用户", bio: "前端独立预览模式", has_avatar: false, stats: { published: 1, likes_received: 12 } });
    if (path === "/api/users/me/profile" && method === "PATCH") return jsonResponse({ user_id: PREVIEW_USER.id, username: PREVIEW_USER.username, display_name: "预览用户", bio: "前端独立预览模式", has_avatar: false, stats: { published: 1, likes_received: 12 } });
    if (path === "/api/users/me/avatar" && method === "POST") return jsonResponse({ user_id: PREVIEW_USER.id, username: PREVIEW_USER.username, display_name: "预览用户", bio: "前端独立预览模式", has_avatar: false, stats: { published: 1, likes_received: 12 } });

    // 预览模式对未使用的 API 保持成功响应，避免某个辅助控件阻塞整个页面。
    return jsonResponse({});
  };
}

export function installPreviewApiMock(): () => void {
  const originalFetch = window.fetch.bind(window);
  window.fetch = createPreviewFetch(originalFetch);
  return () => {
    window.fetch = originalFetch;
  };
}
