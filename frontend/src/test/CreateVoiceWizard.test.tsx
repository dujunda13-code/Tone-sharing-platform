import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DatasetPreprocessResponse } from "../api/types";
import { CreateVoicePage } from "../pages/CreateVoicePage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const READY_HEALTH = { status: "ready", checks: {} };

function postedPaths(): string[] {
  return vi
    .mocked(fetch)
    .mock.calls.filter(([, init]) => (init?.method ?? "GET") === "POST")
    .map(([input]) => new URL(String(input)).pathname);
}

async function walkToReview(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByLabelText("我确认拥有该声音的参考音频授权"));
  await user.upload(
    screen.getByLabelText("授权音频文件"),
    new File(["audio"], "voice.wav", { type: "audio/wav" }),
  );
  await user.click(screen.getByRole("button", { name: "上传并分析" }));
  await screen.findByRole("heading", { name: "音频分析" });
  await user.click(screen.getByRole("button", { name: "开始质检与准备" }));
  await screen.findByText(/自动转写文本一/);
}

describe("CreateVoiceWizard 快速克隆：授权、审核与保存档案", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("shows zero-shot copy and never exposes training wording", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: READY_HEALTH });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);

    expect(screen.getByText(/快速克隆音色/)).toBeInTheDocument();
    expect(screen.getAllByText(/3–10 秒/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/480–720 秒/)).not.toBeInTheDocument();
    expect(screen.queryByText(/开始训练/)).not.toBeInTheDocument();
    expect(screen.queryByText(/epoch/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/损失值/)).not.toBeInTheDocument();
    // 用户可见文案不出现实现与位置词，也不出现磁盘路径。
    expect(document.body).not.toHaveTextContent(/本地|本机/);
    expect(document.body).not.toHaveTextContent(/[A-Za-z]:\\/);
  });

  it("requires reference authorization and saves the profile with exactly one create call", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: READY_HEALTH });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    expect(screen.getByRole("button", { name: "上传并分析" })).toBeDisabled();
    await walkToReview(user);
    await user.type(screen.getByLabelText("音色名称"), "我的旁白");

    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存音色档案" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "保存音色档案" }));
    expect(await screen.findByRole("heading", { name: /创建完成/ })).toBeInTheDocument();

    const posts = postedPaths();
    expect(posts.filter((path) => path === "/api/voices")).toHaveLength(1);
    expect(posts.filter((path) => path.endsWith("/train"))).toHaveLength(0);
  });

  it("requires a voice name and sends it with the create call", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: READY_HEALTH });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();
    // The name is required: the save action stays disabled until it is filled.
    expect(screen.getByRole("button", { name: "保存音色档案" })).toBeDisabled();
    await user.type(screen.getByLabelText("音色名称"), "我的旁白");
    expect(screen.getByRole("button", { name: "保存音色档案" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "保存音色档案" }));
    expect(await screen.findByRole("heading", { name: /创建完成/ })).toBeInTheDocument();

    const createCall = vi
      .mocked(fetch)
      .mock.calls.find(
        ([input, init]) =>
          new URL(String(input)).pathname === "/api/voices" &&
          (init?.method ?? "GET") === "POST",
      );
    expect(createCall).toBeDefined();
    const body = JSON.parse(String(createCall?.[1]?.body)) as {
      dataset_id: string;
      display_name: string;
    };
    expect(body.dataset_id).toBe("dataset-1");
    expect(body.display_name).toBe("我的旁白");

    // 完成页突出音色名称，并提供下一步动作。
    expect(screen.getByText(/我的旁白/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "立即合成" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查看我的音色" })).toBeInTheDocument();

    // 内部模型代号与档案 ID 只能保留在折叠详情里，不出现在正文。
    expect(document.body).not.toHaveTextContent(/V2Pro/i);
    expect(document.body).not.toHaveTextContent(/档案 ID/);
  });

  it("shows a low-SNR reference warning after preprocessing without blocking profile saving", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: READY_HEALTH });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);

    expect(await screen.findByText(/参考音频信噪比较低/)).toBeInTheDocument();
    expect(screen.getByText(/11\.36 dB/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();
    await user.type(screen.getByLabelText("音色名称"), "低噪旁白");
    expect(screen.getByRole("button", { name: "保存音色档案" })).toBeEnabled();
  });

  it("submits manual transcript, emotion and language corrections with a reason", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: READY_HEALTH });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.type(screen.getByLabelText("分段 ID"), "seg_0001");
    await user.type(screen.getByLabelText("校对后文本（留空则仅修改情绪）"), "人工校对后的文本。");
    await user.selectOptions(screen.getByLabelText("修正情绪标签（可选）"), "sad");
    await user.selectOptions(screen.getByLabelText("参考语言（可选）"), "en");
    await user.type(screen.getByLabelText("修改理由（必填）"), "听录修正");
    await user.click(screen.getByRole("button", { name: "保存审核结果" }));

    expect(await screen.findByText("审核结果已保存。")).toBeInTheDocument();
    const reviewCall = vi
      .mocked(fetch)
      .mock.calls.find(([, init]) => init?.method === "PATCH");
    expect(String(reviewCall?.[0])).toContain("/api/datasets/dataset-1/segments/seg_0001");
    const body = JSON.parse(String(reviewCall?.[1]?.body ?? "{}")) as { language?: string };
    expect(body.language).toBe("en");
  });

  it("keeps saving disabled when the reference duration is out of range", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      health: READY_HEALTH,
      preprocessResponse: { dataset_id: "dataset-1", effective_seconds: 2.5, status: "review_required", segment_count: 2, snr_warning_db: 20, warning_codes: [] },
      segmentsResponse: {
        dataset_id: "dataset-1",
        status: "review_required",
        total_segments: 2,
        reviewed_segments: 0,
        effective_seconds: 2.5,
        reviewed_effective_seconds: 0,
        ready_for_profile: false,
        items: [
          {
            segment_id: "seg_0001", order_index: 0,
            relative_path: "datasets/dataset-1/segments/seg_0001.wav",
            duration_seconds: 1.5, language: "zh", split: "reference",
            auto_transcript: "自动转写文本一。", manual_transcript: null,
            effective_transcript: "自动转写文本一。", transcript_source: "auto",
            auto_emotion_label: "happy", auto_emotion_confidence: 0.9,
            manual_emotion_label: null, effective_emotion_label: "happy",
            emotion_source: "auto", review_reason: null, reviewed: false,
          },
          {
            segment_id: "seg_0002", order_index: 1,
            relative_path: "datasets/dataset-1/segments/seg_0002.wav",
            duration_seconds: 1.0, language: "zh", split: "reference",
            auto_transcript: "自动转写文本二。", manual_transcript: null,
            effective_transcript: "自动转写文本二。", transcript_source: "auto",
            auto_emotion_label: "sad", auto_emotion_confidence: 0.62,
            manual_emotion_label: null, effective_emotion_label: "sad",
            emotion_source: "auto", review_reason: null, reviewed: false,
          },
        ],
      },
    });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存音色档案" })).toBeDisabled();
    expect(screen.getByText(/有效时长需在 3–10 秒/)).toBeInTheDocument();
  });

  it("keeps saving disabled while the local service is not ready", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: { status: "not_ready", checks: {} } });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存音色档案" })).toBeDisabled();
    expect(screen.getByText(/服务未就绪/)).toBeInTheDocument();
  });

  it("keeps saving disabled when the dataset is not confirmed", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      health: READY_HEALTH,
      datasets: [{ id: "dataset-1", authorization_confirmed: false, status: "uploaded", effective_seconds: null, asset_count: 1 }],
    });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存音色档案" })).toBeDisabled();
    expect(screen.getByText(/参考音频未完成授权确认/)).toBeInTheDocument();
  });

  it("keeps wizard progress and the review draft when the user leaves and returns", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: READY_HEALTH });
    const { unmount } = render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.type(screen.getByLabelText("分段 ID"), "seg_0001");
    await user.type(screen.getByLabelText("修改理由（必填）"), "听录修正");
    unmount();

    render(<CreateVoicePage onNavigate={vi.fn()} />);

    expect(screen.getByLabelText("我确认拥有该声音的参考音频授权")).toBeChecked();
    expect(screen.getByRole("listitem", { current: "step" })).toHaveTextContent("内容审核");
    expect(screen.getByRole("heading", { name: "内容审核" })).toBeInTheDocument();
    expect(screen.queryByText(/数据集/)).not.toBeInTheDocument();
    expect(screen.getByText(/自动转写文本一/)).toBeInTheDocument();
    expect(screen.getByLabelText("分段 ID")).toHaveValue("seg_0001");
    expect(screen.getByLabelText("修改理由（必填）")).toHaveValue("听录修正");
  });

  it("keeps the running quality check visible when returning while it is in progress", async () => {
    const user = userEvent.setup();
    let resolvePreprocess!: (body: DatasetPreprocessResponse) => void;
    installWorkspaceApiMock({
      health: READY_HEALTH,
      preprocessResponse: new Promise<DatasetPreprocessResponse>((resolve) => {
        resolvePreprocess = resolve;
      }),
    });
    const { unmount } = render(<CreateVoicePage onNavigate={vi.fn()} />);

    await user.click(screen.getByLabelText("我确认拥有该声音的参考音频授权"));
    await user.upload(screen.getByLabelText("授权音频文件"), new File(["audio"], "voice.wav", { type: "audio/wav" }));
    await user.click(screen.getByRole("button", { name: "上传并分析" }));
    await screen.findByRole("heading", { name: "音频分析" });
    await user.click(screen.getByRole("button", { name: "开始质检与准备" }));
    unmount();

    render(<CreateVoicePage onNavigate={vi.fn()} />);

    expect(await screen.findByText(/质检进行中/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "质检中…" })).toBeDisabled();

    resolvePreprocess({ dataset_id: "dataset-1", effective_seconds: 7, status: "review_required", segment_count: 2, snr_warning_db: 20, warning_codes: [] });

    expect(await screen.findByText(/自动转写文本一/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "内容审核" })).toBeInTheDocument();
  });

  it("keeps the prepare failure message when the user leaves and returns", async () => {
    const user = userEvent.setup();
    let rejectPreprocess!: (reason?: unknown) => void;
    installWorkspaceApiMock({
      health: READY_HEALTH,
      preprocessResponse: new Promise((_, reject) => {
        rejectPreprocess = reject;
      }),
    });
    const { unmount } = render(<CreateVoicePage onNavigate={vi.fn()} />);

    await user.click(screen.getByLabelText("我确认拥有该声音的参考音频授权"));
    await user.upload(screen.getByLabelText("授权音频文件"), new File(["audio"], "voice.wav", { type: "audio/wav" }));
    await user.click(screen.getByRole("button", { name: "上传并分析" }));
    await screen.findByRole("heading", { name: "音频分析" });
    await user.click(screen.getByRole("button", { name: "开始质检与准备" }));
    unmount();

    rejectPreprocess(new Error("local service down"));

    render(<CreateVoicePage onNavigate={vi.fn()} />);

    expect(await screen.findByText(/服务暂时不可用/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始质检与准备" })).toBeEnabled();
  });

  it("shows a safe error panel when the upload api fails", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input)).pathname;
      if (path === "/api/auth/me") {
        return new Response(JSON.stringify({ id: "user-1", username: "alice", role: "user", status: "active" }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }
      if (path === "/api/health/ready") {
        return new Response(JSON.stringify(READY_HEALTH), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/datasets" && (init?.method ?? "GET") === "POST") {
        return new Response(
          JSON.stringify({ error: { code: "DATASET_AUTHORIZATION_REQUIRED", message: "需要确认拥有参考音频授权", details: {}, request_id: "r-9" } }),
          { status: 422, headers: { "Content-Type": "application/json" } },
        );
      }
      throw new Error(`unexpected request: ${init?.method ?? "GET"} ${path}`);
    }));
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await user.click(screen.getByLabelText("我确认拥有该声音的参考音频授权"));
    await user.upload(screen.getByLabelText("授权音频文件"), new File(["audio"], "voice.wav", { type: "audio/wav" }));
    await user.click(screen.getByRole("button", { name: "上传并分析" }));

    expect(await screen.findByText(/需要确认拥有参考音频授权/)).toBeInTheDocument();
    expect(screen.queryByText(/Traceback|C:[\\]/)).not.toBeInTheDocument();
  });

  it("shows the auxiliary reference entry on the initial screen before any upload", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({ health: READY_HEALTH });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    // 多段参考入口必须在第一步即可见：用户不必先完成主参考上传、质检与审核才能发现它。
    expect(
      await screen.findByRole("heading", { name: "辅助参考音频（0/4）" }),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("首段参考音频名称（可选）")).not.toBeInTheDocument();
    const addAuxBtn = screen.getByRole("button", { name: /添加辅助参考/ });
    expect(addAuxBtn).toBeEnabled();
    await user.click(addAuxBtn);
    expect(
      await screen.findByRole("heading", { name: "添加辅助参考音频" }),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("参考音频名称（可选）")).not.toBeInTheDocument();
  });

  it("adds an auxiliary reference in the wizard and attaches it when saving the profile", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      health: READY_HEALTH,
      uploadSequence: [
        { dataset_id: "dataset-1", asset_id: "asset-1", content_hash: "hash" },
        { dataset_id: "dataset-aux-1", asset_id: "asset-aux-1", content_hash: "hash-aux" },
      ],
    });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.type(screen.getByLabelText("音色名称"), "我的旁白");
    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();

    const addAuxBtn = await screen.findByRole("button", { name: /添加辅助参考/ });
    expect(addAuxBtn).toBeEnabled();
    await user.click(addAuxBtn);

    const modalHeading = await screen.findByRole("heading", { name: "添加辅助参考音频" });
    const modal = modalHeading.closest(".modal-content") as HTMLElement;
    await user.click(screen.getByLabelText("我确认已获得该音频的合法授权"));
    const auxFileInput = modal.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(auxFileInput, new File(["audio"], "aux.wav", { type: "audio/wav" }));
    await user.click(within(modal).getByRole("button", { name: "上传并分析" }));

    await screen.findByLabelText("参考文本 (请核对并修正)");
    await user.click(within(modal).getByRole("button", { name: "确认添加该参考" }));

    expect(await screen.findByText("aux.wav")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "添加辅助参考音频" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "保存音色档案" }));
    expect(await screen.findByRole("heading", { name: /创建完成/ })).toBeInTheDocument();

    const posts = postedPaths();
    expect(posts.filter((p) => p === "/api/voices")).toHaveLength(1);
    expect(posts.filter((p) => /^\/api\/voices\/[^/]+\/references$/.test(p))).toHaveLength(1);
    const voiceRequests = vi.mocked(fetch).mock.calls.filter(([input, init]) =>
      (new URL(String(input)).pathname === "/api/voices" || /^\/api\/voices\/[^/]+\/references$/.test(new URL(String(input)).pathname)) && init?.method === "POST",
    );
    voiceRequests.forEach(([, init]) => expect(JSON.parse(String(init?.body))).not.toHaveProperty("reference_name"));
  });

  it("removes a queued auxiliary reference before saving so no references call is made", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      health: READY_HEALTH,
      uploadSequence: [
        { dataset_id: "dataset-1", asset_id: "asset-1", content_hash: "hash" },
        { dataset_id: "dataset-aux-1", asset_id: "asset-aux-1", content_hash: "hash-aux" },
      ],
    });
    render(<CreateVoicePage onNavigate={vi.fn()} />);

    await walkToReview(user);
    await user.type(screen.getByLabelText("音色名称"), "我的旁白");
    await user.click(screen.getByRole("button", { name: "全部确认自动结果" }));
    expect(await screen.findByText(/已审核 2 \/ 2 段/)).toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: /添加辅助参考/ }));
    const modal = (await screen.findByRole("heading", { name: "添加辅助参考音频" })).closest(".modal-content") as HTMLElement;
    await user.click(screen.getByLabelText("我确认已获得该音频的合法授权"));
    const auxFileInput = modal.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(auxFileInput, new File(["audio"], "aux.wav", { type: "audio/wav" }));
    await user.click(within(modal).getByRole("button", { name: "上传并分析" }));
    await screen.findByLabelText("参考文本 (请核对并修正)");
    await user.click(within(modal).getByRole("button", { name: "确认添加该参考" }));

    expect(await screen.findByText("aux.wav")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /移除/ }));
    expect(screen.queryByText("aux.wav")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "保存音色档案" }));
    expect(await screen.findByRole("heading", { name: /创建完成/ })).toBeInTheDocument();

    const posts = postedPaths();
    expect(posts.filter((p) => /^\/api\/voices\/[^/]+\/references$/.test(p))).toHaveLength(0);
  });
});
