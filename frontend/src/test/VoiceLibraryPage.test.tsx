import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { VoiceSummary } from "../api/types";
import { VoiceLibraryPage } from "../pages/VoiceLibraryPage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const LOW_SNR_VOICE: VoiceSummary = {
  id: "ready-1",
  display_name: "我的旁白",
  dataset_id: "d1",
  status: "ready",
  mode: "zero_shot",
  base_model_id: "gpt-sovits-v2proplus-official",
  reference_emotions: ["happy", "sad"],
  created_at: "2026-09-12T00:00:00Z",
  can_synthesize: true,
  reference_snr_db: 11.361,
  quality_warning_codes: ["SNR_BELOW_RECOMMENDED"],
};

describe("VoiceLibraryPage 零样本音色列表", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("shows zero-shot mode and reference emotions with the model version collapsed", async () => {
    const onSynthesize = vi.fn();
    installWorkspaceApiMock({
      voices: [LOW_SNR_VOICE],
    });
    render(<VoiceLibraryPage onNavigate={vi.fn()} onSynthesize={onSynthesize} />);

    // The user-named voice leads the card; the raw profile id is never a heading.
    expect(await screen.findByText("我的旁白")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "ready-1" })).not.toBeInTheDocument();
    expect(screen.getByText(/零样本/)).toBeInTheDocument();
    // 内部模型代号不进入正文，只保留在折叠详情中。
    expect(screen.queryByText(/gpt-sovits-v2proplus/)).not.toBeInTheDocument();
    expect(screen.getByText(/高兴、悲伤/)).toBeInTheDocument();
    expect(await screen.findByText(/参考音频信噪比较低/)).toBeInTheDocument();
    expect(screen.getByText(/11\.36 dB/)).toBeInTheDocument();
    expect(screen.getByText(/建议值 20 dB/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "用此音色创作" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: /训练/ })).not.toBeInTheDocument();
  });

  it("keeps historical training-era entries read-only without training actions", async () => {
    installWorkspaceApiMock({
      voices: [
        { id: "legacy-1", display_name: "音色 legacy", dataset_id: "d2", status: "failed", created_at: "2026-09-01T00:00:00Z", can_synthesize: false, mode: null, base_model_id: null, reference_emotions: [] },
      ],
    });
    render(<VoiceLibraryPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

    expect(await screen.findByText("音色 legacy")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /训练/ })).not.toBeInTheDocument();
  });

  it("shows an empty state when the user has no voices", async () => {
    installWorkspaceApiMock({ voices: [] });
    render(<VoiceLibraryPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

    expect(await screen.findByText("还没有音色档案")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "去创建音色" })).toBeInTheDocument();
  });

  it("renders reference count indicator '参考音频 x/5' and lists references", async () => {
    installWorkspaceApiMock({
      voices: [
        {
          ...LOW_SNR_VOICE,
          references: [
            {
              id: "ref-1",
              asset_id: "a1",
              segment_id: "s1",
              prompt_text: "主参考文本",
              prompt_language: "zh",
              emotion_label: "happy",
              is_primary: true,
              created_at: "2026-09-12T00:00:00Z",
            },
            {
              id: "ref-2",
              asset_id: "a2",
              segment_id: "s2",
              prompt_text: "辅助参考文本",
              prompt_language: "zh",
              emotion_label: "sad",
              is_primary: false,
              created_at: "2026-09-12T01:00:00Z",
            },
          ],
        },
      ],
    });
    render(<VoiceLibraryPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

    expect(await screen.findByText("我的旁白")).toBeInTheDocument();
    expect(screen.getByText("参考音频 2/5")).toBeInTheDocument();
  });

  it("adds an auxiliary reference without asking for a separate reference name", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ voices: [LOW_SNR_VOICE] });
    render(<VoiceLibraryPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: "＋ 添加参考" }));
    const modal = screen.getByRole("heading", { name: /追加辅助参考/ }).closest(".modal-content") as HTMLElement;
    await user.upload(modal.querySelector('input[type="file"]') as HTMLInputElement, new File(["audio"], "aux.wav", { type: "audio/wav" }));
    await user.click(within(modal).getByRole("checkbox"));
    await user.click(within(modal).getByRole("button", { name: "上传并分析" }));
    expect(await within(modal).findByRole("button", { name: "确认并追加" })).toBeEnabled();
    expect(within(modal).queryByText("参考音频名称（可选）")).not.toBeInTheDocument();
    await user.click(within(modal).getByRole("button", { name: "确认并追加" }));
    const referenceRequest = vi.mocked(fetch).mock.calls.find(([input, init]) => /^\/api\/voices\/[^/]+\/references$/.test(new URL(String(input)).pathname) && init?.method === "POST");
    expect(referenceRequest).toBeDefined();
    expect(JSON.parse(String(referenceRequest?.[1]?.body))).not.toHaveProperty("reference_name");
  });

  it("shows a safe error panel when the voices api fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input)).pathname;
      if (path === "/api/voices") {
        return new Response(
          JSON.stringify({ error: { code: "HTTP_404", message: "接口暂不可用", details: {}, request_id: "" } }),
          { status: 404, headers: { "Content-Type": "application/json" } },
        );
      }
      throw new Error(`unexpected request: ${init?.method ?? "GET"} ${path}`);
    }));
    render(<VoiceLibraryPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

    expect(await screen.findByRole("button", { name: "重试" })).toBeInTheDocument();
    expect(screen.getByText(/接口暂不可用/)).toBeInTheDocument();
    expect(screen.queryByText(/Traceback|C:[\\]/)).not.toBeInTheDocument();
  });
});
