import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { VoiceSummary } from "../api/types";
import { SynthesisStudioPage } from "../pages/SynthesisStudioPage";
import { synthesisDraftStore } from "../state/drafts";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const LOW_SNR_VOICE: VoiceSummary = {
  id: "ready-1",
  display_name: "温柔旁白",
  dataset_id: "d1",
  status: "ready",
  created_at: "2026-09-06T00:00:00Z",
  can_synthesize: true,
  mode: "zero_shot",
  base_model_id: "gpt-sovits-v2proplus-official",
  reference_emotions: ["happy"],
  reference_snr_db: 11.361,
  quality_warning_codes: ["SNR_BELOW_RECOMMENDED"],
};

const READY_VOICES = [
  { id: "draft-1", display_name: "草稿旁白", dataset_id: "d2", status: "training", created_at: "2026-09-06T00:00:00Z", can_synthesize: false, mode: "zero_shot", base_model_id: "gpt-sovits-v2proplus-official", reference_emotions: ["happy"] },
  LOW_SNR_VOICE,
];

async function choose(user: ReturnType<typeof userEvent.setup>, label: string, option: string) {
  await user.click(await screen.findByRole("combobox", { name: label }));
  await user.click(screen.getByRole("option", { name: option }));
}

describe("SynthesisStudioPage 合成门禁", () => {
  afterEach(() => {
    synthesisDraftStore.reset();
    vi.unstubAllGlobals();
    cleanup();
  });

  it("allows synthesis only with a selected ready voice, text, and literal consent", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ voices: READY_VOICES });
    render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    await screen.findByRole("combobox", { name: "选择音色" });
    await user.click(screen.getByRole("combobox", { name: "选择音色" }));
    expect(screen.getByRole("option", { name: /温柔旁白/ })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /草稿旁白/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: "温柔旁白" }));
    expect(await screen.findByText(/参考音频信噪比较低/)).toBeInTheDocument();
    expect(screen.getByText(/11\.36 dB/)).toBeInTheDocument();
    expect(screen.getByText(/建议值 20 dB/)).toBeInTheDocument();
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeDisabled();
    await user.click(screen.getByLabelText("我确认已获得声音授权"));
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeEnabled();
  });

  it("switches between local WebUI controls and cloud emotion controls", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ voices: READY_VOICES });
    render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    expect(await screen.findByRole("button", { name: "正常合成" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("文本切分方式")).toBeInTheDocument();
    expect(screen.getByLabelText("top_k")).toBeInTheDocument();
    expect(screen.queryByLabelText("情绪控制方式")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "情绪合成" }));

    expect(screen.getByRole("button", { name: "情绪合成" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/ModelVerse IndexTTS-2/)).toBeInTheDocument();
    expect(screen.getByLabelText("情绪控制方式")).toBeInTheDocument();
    expect(screen.getByLabelText("目标情绪")).toBeInTheDocument();
    expect(screen.getByLabelText("输出采样率")).toBeInTheDocument();
    expect(screen.getByLabelText("音量增益")).toBeInTheDocument();
    expect(screen.getByLabelText("随机采样")).toBeInTheDocument();
    expect(screen.getByLabelText("句间静音")).toBeInTheDocument();
    expect(screen.getByLabelText("允许将参考音频发送至云端进行情绪合成")).toBeInTheDocument();
    expect(screen.queryByText(/发音规则/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText("top_k")).not.toBeInTheDocument();
  });

  it("queues the job and only offers audio when the backend marks it verified", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      voices: READY_VOICES,
      syntheses: [{
        job_id: "synthesis-1",
        voice_profile_id: "ready-1",
        text_lang: "zh",
        status: "succeeded",
        download_ready: true,
        watermark_probability: 0.97,
        fingerprint_anomaly: false,
        speaker_similarity: 0.96,
        quality_warning_codes: [],
        created_at: "2026-09-06T00:00:00Z",
      }],
    });
    render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    await choose(user, "选择音色", "温柔旁白");
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    await user.click(screen.getByLabelText("我确认已获得声音授权"));
    await user.click(screen.getByRole("button", { name: "开始安全合成" }));

    expect(await screen.findByText(/正在生成并完成质量检查/)).toBeInTheDocument();
    expect(await screen.findByText(/音色相似度：96\.0%/)).toBeInTheDocument();
    expect(screen.queryByText(/可用但音色还原可能不佳/)).not.toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "下载音频" })).toBeInTheDocument();
    expect(screen.queryByText(/Traceback|C:[\\]/)).not.toBeInTheDocument();
  });

  it("shows the backend candidate progress message while the job runs", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      voices: READY_VOICES,
      syntheses: [{
        job_id: "synthesis-1",
        voice_profile_id: "ready-1",
        text_lang: "zh",
        status: "running",
        download_ready: false,
        watermark_probability: null,
        fingerprint_anomaly: null,
        speaker_similarity: null,
        quality_warning_codes: [],
        progress_message: "正在生成候选 1/3...",
        created_at: "2026-09-16T00:00:00Z",
      }],
    });
    render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    await choose(user, "选择音色", "温柔旁白");
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    await user.click(screen.getByLabelText("我确认已获得声音授权"));
    await user.click(screen.getByRole("button", { name: "开始安全合成" }));

    expect(await screen.findByText("正在生成候选 1/3...")).toBeInTheDocument();
    expect(screen.queryByText("正在生成并完成质量检查，完成后可在此播放或下载。")).not.toBeInTheDocument();
  });

  it("warns on a below-recommendation result while keeping playback and download", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      voices: READY_VOICES,
      syntheses: [{
        job_id: "synthesis-1",
        voice_profile_id: "ready-1",
        text_lang: "zh",
        status: "succeeded",
        download_ready: true,
        watermark_probability: 0.97,
        fingerprint_anomaly: false,
        speaker_similarity: 0.762,
        quality_warning_codes: ["SPEAKER_SIMILARITY_BELOW_RECOMMENDED"],
        created_at: "2026-09-13T00:00:00Z",
      }],
    });
    render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    await choose(user, "选择音色", "温柔旁白");
    // 以名称优先选择音色，值仍为不可变 id。
    expect(screen.getByRole("combobox", { name: "选择音色" })).toHaveTextContent("温柔旁白");
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    await user.click(screen.getByLabelText("我确认已获得声音授权"));
    await user.click(screen.getByRole("button", { name: "开始安全合成" }));

    expect(await screen.findByText(/相似度 76\.2%，低于建议值 90%/)).toBeInTheDocument();
    expect(document.querySelector("audio")).not.toBeNull();
    expect(screen.getByRole("button", { name: "下载音频" })).toBeInTheDocument();
    // 不出现实现词、位置词或裸露的内部任务 ID。
    expect(document.body).not.toHaveTextContent(/GPU|Worker|本机|本地/);
    expect(screen.queryByText(/任务 ID/)).not.toBeInTheDocument();
  });

  it("keeps the creation workspace visible but blocks submission when no ready voice exists", async () => {
    const user = userEvent.setup();
    const navigate = vi.fn();
    synthesisDraftStore.set({ selected: "ready-1" });
    installWorkspaceApiMock({ voices: READY_VOICES.slice(0, 1) });
    render(<SynthesisStudioPage onNavigate={navigate} />);

    expect(await screen.findByText("当前没有可用于创作的音色")).toBeInTheDocument();
    expect(screen.getByLabelText("合成文本")).toBeInTheDocument();
    expect(screen.getByText("合成设置")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "合成结果" })).toBeInTheDocument();
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    await user.click(screen.getByLabelText("我确认已获得声音授权"));
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "去创建音色" }));
    expect(navigate).toHaveBeenCalledWith("create");
  });

  it("keeps the synthesis draft when the user leaves and returns", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ voices: READY_VOICES });
    const { unmount } = render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    await choose(user, "选择音色", "温柔旁白");
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    await user.click(screen.getByLabelText("我确认已获得声音授权"));
    unmount();

    render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    expect(await screen.findByRole("combobox", { name: "选择音色" })).toHaveTextContent("温柔旁白");
    expect(screen.getByLabelText("合成文本")).toHaveValue("你好，世界");
    expect(screen.getByLabelText("我确认已获得声音授权")).toBeChecked();
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeEnabled();
  });

  it("only offers reviewed reference emotions and blocks synthesis without any", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      voices: [
        { id: "ready-1", display_name: "温柔旁白", dataset_id: "d1", status: "ready", created_at: "2026-09-06T00:00:00Z", can_synthesize: true, mode: "zero_shot", base_model_id: "gpt-sovits-v2proplus-official", reference_emotions: ["happy"] },
      ],
    });
    const first = render(<SynthesisStudioPage onNavigate={vi.fn()} />);

    await choose(user, "选择音色", "温柔旁白");
    await choose(user, "情感控制", "手动");

    // Only reviewed reference emotions are offered.
    await user.click(screen.getByRole("combobox", { name: "情感标签" }));
    expect(screen.getByRole("option", { name: "高兴" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "生气" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "悲伤" })).not.toBeInTheDocument();
    first.unmount();

    // A zero-shot voice without any reviewed emotion reference cannot synthesize.
    installWorkspaceApiMock({
      voices: [
        { id: "no-ref", display_name: "无参考音色", dataset_id: "d3", status: "ready", created_at: "2026-09-06T00:00:00Z", can_synthesize: true, mode: "zero_shot", base_model_id: "gpt-sovits-v2proplus-official", reference_emotions: [] },
      ],
    });
    render(<SynthesisStudioPage onNavigate={vi.fn()} />);
    await choose(user, "选择音色", "无参考音色");
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    await choose(user, "情感控制", "手动");
    await user.click(screen.getByLabelText("我确认已获得声音授权"));

    expect(screen.getByText(/该情绪暂无已审核参考/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeDisabled();
  });

  it("keeps a manual voice selection over a stale request when the user leaves and returns", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      voices: [
        ...READY_VOICES,
        { id: "ready-2", display_name: "明亮旁白", dataset_id: "d3", status: "ready", created_at: "2026-09-06T00:00:00Z", can_synthesize: true },
      ],
    });
    const { unmount } = render(
      <SynthesisStudioPage synthesisRequest={{ voiceId: "ready-1", token: 1 }} onNavigate={vi.fn()} />,
    );

    expect(await screen.findByRole("combobox", { name: "选择音色" })).toHaveTextContent("温柔旁白");
    await choose(user, "选择音色", "明亮旁白");
    unmount();

    render(<SynthesisStudioPage synthesisRequest={{ voiceId: "ready-1", token: 1 }} onNavigate={vi.fn()} />);

    expect(await screen.findByRole("combobox", { name: "选择音色" })).toHaveTextContent("明亮旁白");
  });

  it("re-applies the requested voice when it is requested again with a new token", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      voices: [
        ...READY_VOICES,
        { id: "ready-2", display_name: "明亮旁白", dataset_id: "d3", status: "ready", created_at: "2026-09-06T00:00:00Z", can_synthesize: true },
      ],
    });
    const { rerender } = render(
      <SynthesisStudioPage synthesisRequest={{ voiceId: "ready-1", token: 1 }} onNavigate={vi.fn()} />,
    );

    const picker = await screen.findByRole("combobox", { name: "选择音色" });
    expect(picker).toHaveTextContent("温柔旁白");
    await choose(user, "选择音色", "明亮旁白");

    rerender(<SynthesisStudioPage synthesisRequest={{ voiceId: "ready-1", token: 2 }} onNavigate={vi.fn()} />);

    expect(screen.getByRole("combobox", { name: "选择音色" })).toHaveTextContent("温柔旁白");
  });

  it("offers a requested plaza voice with an explicit authorization hint", async () => {
    installWorkspaceApiMock({ voices: [] });
    render(<SynthesisStudioPage synthesisRequest={{ voiceId: "voice-plaza", voiceLabel: "广场音色", token: 1 }} onNavigate={vi.fn()} />);

    expect(await screen.findByRole("combobox", { name: "选择音色" })).toHaveTextContent("广场音色");
    expect(screen.getByText(/正在使用广场音色《广场音色》/)).toBeInTheDocument();
  });

  it("keeps a plaza voice behind the existing consent gate", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ voices: [] });
    render(<SynthesisStudioPage synthesisRequest={{ voiceId: "voice-plaza", voiceLabel: "广场音色", token: 1 }} onNavigate={vi.fn()} />);

    await user.type(await screen.findByLabelText("合成文本"), "你好");
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeDisabled();
  });
});
