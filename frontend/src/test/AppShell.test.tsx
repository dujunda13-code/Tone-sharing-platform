import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { WORKSPACE_NAV } from "../components/AppShell";
import { installWorkspaceApiMock } from "./workspaceApiMock";

describe("AppShell 顶部导航与状态浮层", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("shows only user workspace navigation to a normal user", async () => {
    installWorkspaceApiMock({ currentUser: { role: "user" } });
    render(<App />);

    expect(await screen.findByRole("navigation", { name: "主导航" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "工作台" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "创建音色" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "管理中心" })).not.toBeInTheDocument();
  });

  it("adds the admin management entry only for admins", async () => {
    installWorkspaceApiMock({ currentUser: { role: "admin", username: "admin" } });
    render(<App />);

    expect(await screen.findByRole("navigation", { name: "主导航" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "管理中心" })).toBeInTheDocument();
  });

  it("opens an unobscured readiness dialog with all six checks", async () => {
    const actor = userEvent.setup();
    const okChecks = Object.fromEntries(
      ["database", "storage", "gpu", "gpt_sovits", "emotion2vec", "audioseal"].map(
        (key) => [key, { ok: true, message: "" }],
      ),
    );
    installWorkspaceApiMock({
      currentUser: { role: "user" },
      health: { status: "ready", checks: okChecks },
    });
    render(<App />);

    await actor.click(await screen.findByRole("button", { name: "查看系统状态" }));
    const panel = screen.getByRole("dialog", { name: "系统检查" });
    for (const label of [
      "SQLite 数据库",
      "存储",
      "GPU",
      "GPT-SoVITS",
      "Emotion2Vec",
      "AudioSeal 水印",
    ]) {
      expect(within(panel).getByText(label)).toBeInTheDocument();
    }
    // 状态结论由前端持有，不透传后端实现措辞（全局文案由 Task 11 专项测试收口）。
    expect(panel).not.toHaveTextContent(/本地|本机/);
    expect(within(panel).getAllByText("可用").length).toBeGreaterThan(0);

    await actor.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "系统检查" })).toBeNull();
  });

  it("closes the readiness dialog when navigating", async () => {
    const actor = userEvent.setup();
    installWorkspaceApiMock({ currentUser: { role: "user" } });
    render(<App />);

    await actor.click(await screen.findByRole("button", { name: "查看系统状态" }));
    expect(screen.getByRole("dialog", { name: "系统检查" })).toBeInTheDocument();

    await actor.click(screen.getByRole("button", { name: "任务中心" }));
    expect(screen.queryByRole("dialog", { name: "系统检查" })).toBeNull();
  });

  it("workspace nav places 音色广场 second", () => {
    const labels = WORKSPACE_NAV.map((item) => item.label);
    expect(labels[0]).toBe("工作台");
    expect(labels[1]).toBe("音色广场");
    expect(labels).toContain("我的音色");
  });

  it("provides a mobile menu button controlling the collapsed navigation", async () => {
    const actor = userEvent.setup();
    installWorkspaceApiMock({ currentUser: { role: "user" } });
    render(<App />);

    const toggle = await screen.findByRole("button", { name: "打开导航菜单" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("navigation", { name: "移动端导航" })).toBeNull();
    await actor.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("navigation", { name: "移动端导航" })).toBeVisible();
    expect(
      within(screen.getByRole("navigation", { name: "移动端导航" })).getByRole("button", {
        name: "语音创作",
      }),
    ).toBeInTheDocument();
  });
});
