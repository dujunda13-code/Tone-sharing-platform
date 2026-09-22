import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";

import { NotificationBell } from "../components/NotificationBell";
import { installWorkspaceApiMock } from "./workspaceApiMock";

afterEach(() => { vi.unstubAllGlobals(); cleanup(); });

test("shows an unread notification label only when local unread count is positive", async () => {
  installWorkspaceApiMock({ unreadCount: 3 });
  render(<NotificationBell onViewAll={vi.fn()} />);
  expect(await screen.findByLabelText("通知（3 条未读）")).toBeInTheDocument();
});

test("opens local notifications, marks them read and opens full history", async () => {
  const fetchMock = installWorkspaceApiMock({ unreadCount: 1, notifications: [{ id: "n1", type: "like", actor_display_name: "bob", post_id: "p1", voice_name: "零样本演示音色", comment_excerpt: null, is_read: false, created_at: "2026-09-17T00:00:00Z" }] });
  const onViewAll = vi.fn();
  const user = userEvent.setup();
  render(<NotificationBell onViewAll={onViewAll} />);

  await user.click(await screen.findByLabelText("通知（1 条未读）"));
  expect(await screen.findByText(/bob 赞了你的音色《零样本演示音色》/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "全部已读" }));
  await user.click(screen.getByRole("button", { name: "查看全部" }));
  expect(onViewAll).toHaveBeenCalledTimes(1);
  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(calls.some(([url, init]) => String(url).includes("/api/notifications/mark-read") && init?.method === "POST")).toBe(true);
});
