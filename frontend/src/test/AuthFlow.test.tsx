import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { installWorkspaceApiMock } from "./workspaceApiMock";

describe("authentication flow", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("does not mount the workspace before login", async () => {
    installWorkspaceApiMock({ authenticated: false });
    render(<App />);

    const loginRegion = await screen.findByRole("region", { name: "登录" });
    expect(within(loginRegion).getByRole("heading", { name: "登录账号" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "工作台" })).not.toBeInTheDocument();
    // 用户可见界面不出现实现措辞、绝对路径或原始请求 ID。
    expect(document.body.textContent).not.toMatch(/本地|本机/);
    expect(document.body.textContent).not.toMatch(/[A-Za-z]:\\|request_id/);
  });

  it("registers, logs in, mounts the workspace, and logs out", async () => {
    installWorkspaceApiMock({ authenticated: false });
    const actor = userEvent.setup();
    render(<App />);
    // 登录主操作在前，文字注册链接位于其后。
    const loginRegion = await screen.findByRole("region", { name: "登录" });
    const loginButton = within(loginRegion).getByRole("button", { name: "登录" });
    const registerLink = within(loginRegion).getByRole("button", { name: "立即注册" });
    expect(loginButton.compareDocumentPosition(registerLink) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    await actor.click(registerLink);
    expect(await screen.findByRole("heading", { name: "创建账号" })).toBeInTheDocument();
    expect(screen.getByText("至少 8 位；密码不会保存在浏览器存储。")).toBeInTheDocument();
    await actor.type(screen.getByLabelText("用户名"), "alice");
    await actor.type(screen.getByLabelText("密码"), "eight888");
    await actor.type(screen.getByLabelText("确认密码"), "eight888");
    // 密码可见性可重复切换（新页面内置 PasswordField）。
    await actor.click(screen.getByRole("button", { name: "显示密码" }));
    expect(screen.getByLabelText("密码")).toHaveAttribute("type", "text");
    await actor.click(screen.getByRole("button", { name: "隐藏密码" }));
    expect(screen.getByLabelText("密码")).toHaveAttribute("type", "password");
    await actor.click(screen.getByRole("button", { name: "注册" }));

    const backToLogin = await screen.findByRole("region", { name: "登录" });
    expect(
      within(backToLogin).getByRole("heading", { name: "登录账号" }),
    ).toBeInTheDocument();
    await actor.type(within(backToLogin).getByLabelText("用户名"), "alice");
    await actor.type(within(backToLogin).getByLabelText("密码"), "eight888");
    await actor.click(within(backToLogin).getByRole("button", { name: "登录" }));

    expect(await screen.findByRole("button", { name: "工作台" })).toBeInTheDocument();
    expect(screen.getByText("alice")).toBeInTheDocument();
    await actor.click(screen.getByRole("button", { name: "退出登录" }));
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "欢迎回来" })).toBeInTheDocument(),
    );
  });
});
