import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { LoginPage } from "../pages/LoginPage";
import { RegisterPage } from "../pages/RegisterPage";

describe("认证页预览图视觉契约", () => {
  afterEach(() => cleanup());

  it("登录页使用与工作台一致的全屏氛围布局和品牌元素", () => {
    render(<LoginPage onLogin={vi.fn()} onRegister={vi.fn()} />);

    const page = screen.getByRole("main");
    expect(page).toHaveClass("auth-preview", "auth-preview-login");
    const artPanel = page.querySelector(".auth-art-panel");
    expect(artPanel).toBeInTheDocument();
    expect(page.querySelector(".auth-brand-mark")).toBeInTheDocument();
    expect(page.querySelector(".auth-brand-mark .brand-glyph")).toBeInTheDocument();
    expect(page.querySelector("video.auth-preview-ribbon source[src='/dashboard-ribbon-motion.mp4']")).toBeInTheDocument();
    expect(page.querySelector(".password-toggle svg")).toBeInTheDocument();

    const form = screen.getByRole("region", { name: "登录" });
    expect(form).toHaveClass("auth-card");
    expect(within(artPanel as HTMLElement).getByRole("heading", { name: "欢迎回来" })).toBeInTheDocument();
    expect(within(artPanel as HTMLElement).queryByText("让每一种声音，都能被清晰表达。")).not.toBeInTheDocument();
    expect(artPanel?.querySelector(".auth-art-signature")).toBeInTheDocument();
    expect(within(form).getByRole("heading", { name: "登录账号" })).toBeInTheDocument();
    expect(form.querySelector(".auth-divider")).not.toBeInTheDocument();
    expect(form.querySelector(".auth-provider-row")).not.toBeInTheDocument();
    expect(within(form).getByLabelText("用户名")).toHaveAttribute("type", "text");
    expect(within(form).getByRole("button", { name: "登录" })).toHaveClass("primary");
    expect(within(form).getByText("还没有账号？")).toBeInTheDocument();
    expect(within(form).getByRole("button", { name: "立即注册" })).toHaveClass("auth-register-link");
    expect(document.body.textContent).not.toMatch(/本地|本机/);
  });

  it("注册页复用相同的品牌背景、图标和表单卡片", () => {
    render(<RegisterPage onRegistered={vi.fn()} onBack={vi.fn()} />);

    const page = screen.getByRole("main");
    expect(page).toHaveClass("auth-preview", "auth-preview-register");
    const artPanel = page.querySelector(".auth-art-panel");
    expect(artPanel).toBeInTheDocument();
    expect(page.querySelector(".auth-brand-mark")).toBeInTheDocument();
    expect(page.querySelector(".auth-brand-mark .brand-glyph")).toBeInTheDocument();
    const form = screen.getByRole("region", { name: "注册" });
    expect(form).toHaveClass("auth-card");
    expect(within(artPanel as HTMLElement).getByRole("heading", { name: "创建账号" })).toBeInTheDocument();
    expect(artPanel?.querySelector(".auth-art-signature")).toBeInTheDocument();
    expect(within(form).getByRole("heading", { name: "注册新账号" })).toBeInTheDocument();
    expect(within(form).getByLabelText("用户名")).toHaveAttribute("type", "text");
    expect(within(form).getByRole("button", { name: "显示确认密码" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "注册" })).toHaveClass("primary");
  });

  it("确认密码可以独立切换显示状态", async () => {
    const user = userEvent.setup();
    render(<RegisterPage onRegistered={vi.fn()} onBack={vi.fn()} />);
    const confirmation = screen.getByLabelText("确认密码", { exact: true });
    expect(confirmation).toHaveAttribute("type", "password");
    await user.click(screen.getByRole("button", { name: "显示确认密码" }));
    expect(confirmation).toHaveAttribute("type", "text");
    expect(screen.getByLabelText("密码", { exact: true })).toHaveAttribute("type", "password");
    await user.click(screen.getByRole("button", { name: "隐藏确认密码" }));
    expect(confirmation).toHaveAttribute("type", "password");
  });
});
