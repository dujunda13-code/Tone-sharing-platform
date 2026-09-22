import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { formatDuration } from "../components/TaskTimeline";
import { TaskCenterPage } from "../pages/TaskCenterPage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

describe("formatDuration", () => {
  it("formats elapsed seconds and the pending state", () => {
    expect(formatDuration(null, null)).toBe("尚未开始");
    expect(formatDuration("2026-09-06T00:00:00Z", "2026-09-06T00:00:42Z")).toBe("42 秒");
  });
});

describe("TaskCenterPage 任务中心", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("lists real task summaries in natural language with details collapsed", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      jobs: [
        {
          id: "job-1",
          kind: "synthesize",
          status: "failed",
          error_code: "SYNTHESIS_PIPELINE_UNAVAILABLE",
          public_message: "本机合成前置条件未满足",
          created_at: "2026-09-06T00:00:00Z",
          started_at: "2026-09-06T00:00:10Z",
          finished_at: "2026-09-06T00:00:52Z",
        },
        {
          id: "job-2",
          kind: "train",
          status: "running",
          error_code: null,
          public_message: null,
          created_at: "2026-09-06T00:00:00Z",
          started_at: "2026-09-06T00:00:01Z",
          finished_at: null,
        },
      ],
    });
    render(<TaskCenterPage />);

    // 失败结论以自然语言呈现，实现词替换为用户表述；等待中任务不暴露 Worker 等实现词。
    expect(await screen.findByText("系统合成前置条件未满足")).toBeInTheDocument();
    expect(screen.getByText(/正在等待系统处理/)).toBeInTheDocument();
    expect(screen.getByText(/耗时 42 秒/)).toBeInTheDocument();
    expect(document.body).not.toHaveTextContent(/本机|本地|Worker/);
    // 内部任务 ID 与错误代码默认折叠，点击"查看详情"后才可见。
    expect(screen.queryByText("job-1")).not.toBeInTheDocument();
    expect(screen.queryByText(/SYNTHESIS_PIPELINE_UNAVAILABLE/)).not.toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "查看详情" })[0]);
    expect(screen.getByText(/SYNTHESIS_PIPELINE_UNAVAILABLE/)).toBeVisible();
    // 任务类型使用用户可理解的标签；历史训练记录仅作只读展示。
    expect(screen.getByText("语音合成")).toBeInTheDocument();
    expect(screen.getByText("历史训练")).toBeInTheDocument();
    expect(screen.queryByText(/Traceback|C:[\\]|payload/)).not.toBeInTheDocument();
  });

  it("filters tasks by status using the backend query", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      jobs: [
        { id: "job-1", kind: "train", status: "running", error_code: null, public_message: null, created_at: "2026-09-06T00:00:00Z", started_at: null, finished_at: null },
      ],
    });
    render(<TaskCenterPage />);

    await screen.findByText("历史训练");
    await user.click(screen.getByRole("button", { name: "执行中" }));
    expect(await screen.findByText("历史训练")).toBeInTheDocument();
  });

  it("shows an empty state when there are no tasks", async () => {
    installWorkspaceApiMock({ jobs: [] });
    render(<TaskCenterPage />);

    expect(await screen.findByText("还没有任务")).toBeInTheDocument();
  });

  it("keeps the status filter when the user leaves and returns", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      jobs: [
        { id: "job-1", kind: "train", status: "running", error_code: null, public_message: null, created_at: "2026-09-06T00:00:00Z", started_at: null, finished_at: null },
      ],
    });
    const { unmount } = render(<TaskCenterPage />);

    await screen.findByText("历史训练");
    await user.click(screen.getByRole("button", { name: "执行中" }));
    unmount();

    render(<TaskCenterPage />);

    expect(screen.getByRole("button", { name: "执行中" })).toHaveClass("active-filter");
  });
});
