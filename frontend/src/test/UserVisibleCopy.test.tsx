import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { installWorkspaceApiMock } from "./workspaceApiMock";

/**
 * 全局用户可见文案契约：登录、用户工作台与管理员工作台在任何页面都
 * 不得出现实现/位置词、磁盘路径、request_id 或 Traceback。
 */
const FORBIDDEN_PATTERNS = [
  /本地|本机/,
  /[A-Za-z]:\\|\/(?:Users|home|var|tmp)\//,
  /request_id|Traceback/,
];

const READY_HEALTH = { status: "ready", checks: {} };

const USER_WORKSPACE = {
  currentUser: { id: "user-1", username: "alice", role: "user" as const },
  health: READY_HEALTH,
  dashboard: {
    counts: { datasets: 1, voices: 2, jobs: 3, active_jobs: 1, syntheses: 2 },
    recent_jobs: [
      {
        id: "job-1",
        kind: "synthesize",
        status: "failed",
        error_code: "SYNTHESIS_PIPELINE_UNAVAILABLE",
        public_message: "本机合成前置条件未满足",
        created_at: "2026-09-06T00:00:00Z",
      },
    ],
    voices: [],
    readiness: { status: "ready", checks: {} },
  },
  voices: [
    {
      id: "voice-1",
      display_name: "我的旁白",
      dataset_id: "dataset-1",
      status: "ready",
      mode: "zero_shot",
      base_model_id: "gpt-sovits-v2proplus-official",
      reference_emotions: ["happy"],
      quality_warning_codes: [],
      can_synthesize: true,
      created_at: "2026-09-06T00:00:00Z",
    },
  ],
  jobs: [],
  syntheses: [
    {
      job_id: "synthesis-1",
      voice_profile_id: "voice-1",
      text_lang: "zh",
      status: "succeeded",
      download_ready: true,
      watermark_probability: 0.97,
      fingerprint_anomaly: false,
      speaker_similarity: 0.96,
      quality_warning_codes: [],
      created_at: "2026-09-06T00:00:00Z",
    },
  ],
};

const ADMIN_WORKSPACE = {
  currentUser: { id: "admin-1", username: "admin", role: "admin" as const },
  health: READY_HEALTH,
  dashboard: {
    counts: { datasets: 0, voices: 0, jobs: 0, active_jobs: 0, syntheses: 0 },
    recent_jobs: [],
    voices: [],
    readiness: { status: "ready", checks: {} },
  },
  adminUsers: [
    { id: "admin-1", username: "admin", role: "admin", status: "active", created_at: "2026-09-06T00:00:00Z", dataset_count: 0, voice_count: 0, job_count: 0 },
  ],
  auditEvents: [
    { id: 1, event_type: "admin.user_status_changed", subject_id: "u-9", metadata: { old_status: "active", new_status: "disabled" }, created_at: "2026-09-06T00:00:02Z" },
  ],
  auditNextBeforeId: null,
};

function expectCleanBody(): void {
  for (const pattern of FORBIDDEN_PATTERNS) {
    expect(document.body.textContent).not.toMatch(pattern);
  }
}

describe("UserVisibleCopy 全局用户可见文案契约", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("keeps the login surface free of implementation wording", async () => {
    installWorkspaceApiMock({ authenticated: false });
    render(<App />);

    expect(await screen.findByRole("button", { name: "登录" })).toBeInTheDocument();
    expectCleanBody();
  });

  it("keeps every user workspace page free of implementation wording", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ ...USER_WORKSPACE });
    render(<App />);

    expect(await screen.findByText(/让每一种声音/)).toBeInTheDocument();
    expectCleanBody();

    for (const [page, marker] of [
      ["我的音色", /我的旁白/],
      ["创建音色", /快速克隆音色/],
      ["语音创作", /开始安全合成/],
      ["任务中心", /音频分析和语音合成任务/],
      ["安全中心", /嵌入来源水印/],
    ] as const) {
      await user.click(screen.getByRole("button", { name: page }));
      expect(await screen.findByText(marker)).toBeInTheDocument();
      expectCleanBody();
    }
  });

  it("keeps the admin workspace free of implementation wording", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ ...ADMIN_WORKSPACE });
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "管理中心" }));
    expect(await screen.findByRole("table", { name: "用户列表" })).toBeInTheDocument();
    expectCleanBody();
  });
});
