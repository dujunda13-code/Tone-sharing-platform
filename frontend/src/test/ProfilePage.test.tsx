import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";

import { ProfilePage } from "../pages/ProfilePage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const PROFILE = {
  user_id: "user-1",
  username: "alice",
  display_name: "小艾",
  bio: "爱配音",
  has_avatar: false,
  stats: { published: 2, likes_received: 5 },
};

afterEach(() => {
  vi.unstubAllGlobals();
  cleanup();
});

test("shows a profile header with local activity tabs", async () => {
  installWorkspaceApiMock({ userProfile: PROFILE });
  render(<ProfilePage tab="posts" onTabChange={vi.fn()} onNavigate={vi.fn()} />);

  const nickname = await screen.findByLabelText("昵称");
  expect(nickname).toHaveValue("小艾");
  expect(nickname.closest(".profile-edit-panel")).toBeInTheDocument();
  expect(screen.getByText("alice")).toBeInTheDocument();
  expect(screen.getByText("alice").closest(".profile-identity-panel")).toBeInTheDocument();
  expect(screen.getByText(/发布 2/)).toBeInTheDocument();
  expect(screen.getByText(/获赞 5/)).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "我的发布" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "我的收藏" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "我的评论" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "通知历史" })).toBeInTheDocument();
  expect(screen.getByRole("tablist", { name: "个人活动" }).closest(".profile-activity-card")).toBeInTheDocument();
  expect(screen.queryByText("我的空间")).not.toBeInTheDocument();
});

test("shows a real empty publication state with a create action", async () => {
  installWorkspaceApiMock({ userProfile: PROFILE, plazaPosts: [] });
  const onNavigate = vi.fn();
  const user = userEvent.setup();
  render(<ProfilePage tab="posts" onTabChange={vi.fn()} onNavigate={onNavigate} />);

  expect(await screen.findByText("还没有发布过音色")).toBeInTheDocument();
  expect(screen.queryByText("零样本演示音色")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "去创建音色" }));
  expect(onNavigate).toHaveBeenCalledWith("create");
});

test("saves a changed profile name through the local profile endpoint", async () => {
  const fetchMock = installWorkspaceApiMock({ userProfile: PROFILE });
  const user = userEvent.setup();
  render(<ProfilePage tab="posts" onTabChange={vi.fn()} onNavigate={vi.fn()} />);

  const nameInput = await screen.findByLabelText("昵称");
  await user.clear(nameInput);
  await user.type(nameInput, "新昵称");
  await user.click(screen.getByRole("button", { name: "保存资料" }));

  await waitFor(() => {
    const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
    expect(calls.some(([url, init]) => String(url).includes("/api/users/me/profile") && init?.method === "PATCH" && String(init.body).includes("新昵称"))).toBe(true);
  });
});

test("disables profile saving when nickname exceeds 32 characters", async () => {
  installWorkspaceApiMock({ userProfile: PROFILE });
  const user = userEvent.setup();
  render(<ProfilePage tab="posts" onTabChange={vi.fn()} onNavigate={vi.fn()} />);

  const nameInput = await screen.findByLabelText("昵称");
  fireEvent.change(nameInput, { target: { value: "长".repeat(33) } });
  expect(screen.getByRole("button", { name: "保存资料" })).toBeDisabled();
});

test("uploads a supported avatar through multipart form data", async () => {
  const fetchMock = installWorkspaceApiMock({ userProfile: PROFILE });
  const user = userEvent.setup();
  render(<ProfilePage tab="posts" onTabChange={vi.fn()} onNavigate={vi.fn()} />);

  const file = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], "avatar.png", { type: "image/png" });
  await user.upload((await screen.findByLabelText("上传头像")) as HTMLInputElement, file);
  await waitFor(() => {
    const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
    expect(calls.some(([url, init]) => String(url).includes("/api/users/me/avatar") && init?.method === "POST")).toBe(true);
  });
});

test("lists notifications and marks all notifications read", async () => {
  const fetchMock = installWorkspaceApiMock({
    userProfile: PROFILE,
    notifications: [{ id: "n1", type: "like", actor_display_name: "bob", post_id: "p1", voice_name: "零样本演示音色", comment_excerpt: null, is_read: false, created_at: "2026-09-17T00:00:00Z" }],
  });
  const user = userEvent.setup();
  render(<ProfilePage tab="notifications" onTabChange={vi.fn()} onNavigate={vi.fn()} />);

  expect(await screen.findByText(/bob 赞了你的音色《零样本演示音色》/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "全部已读" }));
  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(calls.some(([url, init]) => String(url).includes("/api/notifications/mark-read") && init?.method === "POST")).toBe(true);
});

test("switching tabs fetches the requested local history", async () => {
  const fetchMock = installWorkspaceApiMock({ userProfile: PROFILE, myComments: [] });
  const user = userEvent.setup();
  const onTabChange = vi.fn();
  render(<ProfilePage tab="posts" onTabChange={onTabChange} onNavigate={vi.fn()} />);

  await screen.findByLabelText("昵称");
  await user.click(screen.getByRole("tab", { name: "我的评论" }));
  expect(onTabChange).toHaveBeenCalledWith("comments");
  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(calls.some(([url]) => String(url).includes("mine=true"))).toBe(true);
});
