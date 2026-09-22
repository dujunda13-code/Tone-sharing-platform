import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { api } from "../api/client";
import { DashboardPage } from "../pages/DashboardPage";
import { installWorkspaceApiMock, ZERO_SHOT_VOICE } from "./workspaceApiMock";

const READY_DASHBOARD = {
  counts: { datasets: 2, voices: 1, jobs: 5, active_jobs: 1, syntheses: 4 },
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
  voices: [ZERO_SHOT_VOICE],
  readiness: { status: "ready", checks: {} },
};

describe("DashboardPage 编辑式看板", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("renders the editorial hero, four approved stats, and a green system status", async () => {
    installWorkspaceApiMock({ dashboard: READY_DASHBOARD });
    render(<App />);

    expect(
      await screen.findByText(/让每一种声音，都能被清晰表达/),
    ).toBeInTheDocument();
    const heroWaves = document.querySelector(".dashboard-hero-waves");
    expect(heroWaves).toBeInTheDocument();
    expect(heroWaves?.querySelectorAll("path")).toHaveLength(3);
    const stats = screen.getByRole("region", { name: "资源概览" });
    expect(within(stats).getByText("参考音频")).toBeInTheDocument();
    expect(within(stats).getByText("我的音色")).toBeInTheDocument();
    expect(within(stats).getByText("处理中任务")).toBeInTheDocument();
    expect(within(stats).getByText("合成作品")).toBeInTheDocument();
    expect(within(stats).getByText("2")).toBeInTheDocument();
    expect(within(stats).getAllByText("1").length).toBeGreaterThanOrEqual(1);
    expect(within(stats).getByText("4")).toBeInTheDocument();
    // 系统状态：可用时绿灯常亮；不再出现"本机就绪状态"。
    expect(screen.getByText("系统状态")).toBeInTheDocument();
    const light = screen.getByText("系统可用");
    expect(light.closest(".ready-light")).toHaveClass("ok");
    expect(screen.queryByText("本机就绪状态")).not.toBeInTheDocument();
    // 用户可见文案不出现"数据集""本地""本机"或内部 ID。
    expect(document.body).not.toHaveTextContent(/数据集|本地|本机/);
    expect(document.body).not.toHaveTextContent(/voice-1|dataset-1/);
  });

  it("shows voice rows headed by the user-named voice", async () => {
    installWorkspaceApiMock({ dashboard: READY_DASHBOARD });
    render(<App />);

    expect(await screen.findByText("零样本演示音色")).toBeInTheDocument();
    const voiceSection = screen.getByRole("region", { name: "我的音色" });
    expect(
      within(voiceSection).getByText("零样本演示音色"),
    ).toBeInTheDocument();
    expect(within(voiceSection).queryByText("voice-1")).toBeNull();
  });

  it("maps backend job messages to user-facing wording", async () => {
    installWorkspaceApiMock({ dashboard: READY_DASHBOARD });
    render(<App />);

    expect(await screen.findByText("系统合成前置条件未满足")).toBeInTheDocument();
    expect(screen.queryByText("本机合成前置条件未满足")).not.toBeInTheDocument();
    expect(screen.queryByText(/Traceback|C:[\\]|训练完成/)).not.toBeInTheDocument();
  });

  it("describes the zero-shot quick-clone flow instead of any training", async () => {
    installWorkspaceApiMock({
      dashboard: {
        counts: { datasets: 0, voices: 0, jobs: 0, active_jobs: 0, syntheses: 0 },
        recent_jobs: [],
        voices: [],
        readiness: { status: "ready", checks: {} },
      },
    });
    render(<App />);

    expect(await screen.findByText(/零样本/)).toBeInTheDocument();
    expect(screen.queryByText(/开始训练/)).not.toBeInTheDocument();
    expect(screen.queryByText(/训练与安全合成/)).not.toBeInTheDocument();
  });

  it("shows counts, empty states and safe retry when the dashboard api fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input)).pathname;
      if (path === "/api/auth/me") {
        return new Response(JSON.stringify({ id: "user-1", username: "alice", role: "user", status: "active" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (path === "/api/health/ready") {
        return new Response(JSON.stringify({ status: "not_ready", checks: {} }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (path === "/api/dashboard") {
        return new Response(JSON.stringify({ error: { code: "HTTP_404", message: "接口暂不可用", details: {}, request_id: "r-1" } }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`unexpected request: ${init?.method ?? "GET"} ${path}`);
    }));
    render(<App />);

    expect(await screen.findByRole("button", { name: "重试" })).toBeInTheDocument();
    expect(screen.getByText(/接口暂不可用/)).toBeInTheDocument();
    expect(screen.queryByText(/Traceback|C:[\\]/)).not.toBeInTheDocument();
  });

  it("loads its overview without issuing a second readiness request", async () => {
    const dashboard = vi.spyOn(api, "dashboard").mockResolvedValue({
      counts: { datasets: 0, voices: 0, jobs: 0, active_jobs: 0, syntheses: 0 },
      recent_jobs: [],
      voices: [],
      readiness: { status: "ready", checks: {} },
    });
    const health = vi.spyOn(api, "health").mockResolvedValue({ status: "ready", checks: {} });

    render(<DashboardPage onNavigate={vi.fn()} />);

    expect(
      await screen.findByText(/让每一种声音，都能被清晰表达/),
    ).toBeInTheDocument();
    expect(dashboard).toHaveBeenCalledTimes(1);
    expect(health).not.toHaveBeenCalled();
  });
});
