import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SafetyCenterPage } from "../pages/SafetyCenterPage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const STAGE_ORDER = /授权确认[\s\S]*内容检查[\s\S]*来源水印[\s\S]*声音指纹[\s\S]*结果判定/;

describe("SafetyCenterPage 安全中心", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("presents the safety pipeline as ordered linear stages without a card grid", async () => {
    installWorkspaceApiMock({
      syntheses: [
        {
          job_id: "verified-1",
          voice_profile_id: "ready-1",
          text_lang: "zh",
          status: "succeeded",
          download_ready: true,
          watermark_probability: 0.98,
          fingerprint_anomaly: false,
          speaker_similarity: 0.96,
          quality_warning_codes: [],
          created_at: "2026-09-06T00:00:00Z",
        },
        {
          job_id: "blocked-1",
          voice_profile_id: "ready-1",
          text_lang: "zh",
          status: "failed",
          download_ready: false,
          watermark_probability: null,
          fingerprint_anomaly: null,
          speaker_similarity: null,
          quality_warning_codes: [],
          created_at: "2026-09-06T00:00:00Z",
        },
      ],
    });
    render(<SafetyCenterPage />);

    // 五个安全阶段按固定顺序线性呈现，而不是重复小卡片的网格。
    const stages = await screen.findByRole("list", { name: "安全验证流程" });
    const joined = within(stages)
      .getAllByRole("listitem")
      .map((item) => item.textContent ?? "")
      .join("\n");
    expect(joined).toMatch(STAGE_ORDER);
    expect(document.querySelector(".page-grid")).toBeNull();
    // 内部任务 ID 不再作为记录标题出现。
    expect(screen.queryByText("verified-1")).not.toBeInTheDocument();
  });

  it("hides download for unpublished results and explains unconfirmed origin honestly", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      syntheses: [
        {
          job_id: "blocked-1",
          voice_profile_id: "ready-1",
          text_lang: "zh",
          status: "failed",
          download_ready: false,
          watermark_probability: null,
          fingerprint_anomaly: null,
          speaker_similarity: null,
          quality_warning_codes: [],
          created_at: "2026-09-06T00:00:00Z",
        },
      ],
      watermarkVerification: { job_id: "blocked-1", probability: 0.01, payload: null, payload_matches_job: false },
    });
    render(<SafetyCenterPage />);

    expect(await screen.findByRole("button", { name: "复检来源" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "下载音频" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "复检来源" }));
    expect(await screen.findAllByText(/无法确认来源/)).not.toHaveLength(0);
    expect(screen.queryByText("平台生成")).not.toBeInTheDocument();
  });

  it("marks verified outputs as traceable and offers download", async () => {
    installWorkspaceApiMock({
      syntheses: [
        {
          job_id: "verified-1",
          voice_profile_id: "ready-1",
          text_lang: "zh",
          status: "succeeded",
          download_ready: true,
          watermark_probability: 0.98,
          fingerprint_anomaly: false,
          speaker_similarity: 0.96,
          quality_warning_codes: [],
          created_at: "2026-09-06T00:00:00Z",
        },
      ],
      watermarkVerification: { job_id: "verified-1", probability: 0.98, payload: 1, payload_matches_job: true },
    });
    render(<SafetyCenterPage />);

    expect(await screen.findByRole("button", { name: "下载音频" })).toBeInTheDocument();
    expect(screen.getByText(/水印校验通过/)).toBeInTheDocument();
  });

  it("shows an empty state when nothing was synthesized yet", async () => {
    installWorkspaceApiMock({ syntheses: [] });
    render(<SafetyCenterPage />);

    expect(await screen.findByText("还没有合成记录")).toBeInTheDocument();
  });
});
