import { afterEach, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const PLAZA_POST = {
  id: "post-1",
  voice: {
    voice_profile_id: "voice-1",
    display_name: "零样本演示音色",
    status: "ready",
    reference_count: 2,
    reference_emotions: ["happy"],
  },
  author: { user_id: "user-1", display_name: "alice", has_avatar: false },
  description: "简介",
  like_count: 1,
  comment_count: 2,
  favorite_count: 3,
  liked_by_me: false,
  favorited_by_me: true,
  created_at: "2026-09-17T00:00:00Z",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

it("api.plazaPosts resolves mocked list", async () => {
  installWorkspaceApiMock({ plazaPosts: [PLAZA_POST] });
  const list = await api.plazaPosts({ mine: true });
  expect(list.total).toBe(1);
  expect(list.items[0].voice.display_name).toBe("零样本演示音色");
});

it("api.plazaPublish posts to /api/plaza/posts", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [] });
  await api.plazaPublish({ voice_profile_id: "voice-1" });
  const calls = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls;
  const published = calls.find(([, init]) => String((init as RequestInit | undefined)?.method) === "POST");
  expect(published?.[0]).toContain("/api/plaza/posts");
});

it("api.notificationsUnreadCount and markRead hit notification routes", async () => {
  const fetchMock = installWorkspaceApiMock({ unreadCount: 2 });
  const count = await api.notificationsUnreadCount();
  await api.notificationsMarkRead({ all: true });
  const calls = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls;
  expect(count.count).toBe(2);
  expect(calls.some(([url]) => String(url).includes("/api/notifications/unread-count"))).toBe(true);
  expect(calls.some(([url, init]) => String(url).includes("/api/notifications/mark-read") && String((init as RequestInit | undefined)?.method) === "POST")).toBe(true);
});

it("api.userProfile resolves mocked profile", async () => {
  installWorkspaceApiMock({
    userProfile: {
      user_id: "user-1",
      username: "alice",
      display_name: "小雅",
      bio: null,
      has_avatar: false,
      stats: { published: 1, likes_received: 2 },
    },
  });
  const profile = await api.userProfile();
  expect(profile.display_name).toBe("小雅");
  expect(profile.stats.likes_received).toBe(2);
});

it("api plaza download URLs include API_BASE and post id", () => {
  expect(api.plazaDownloadUrl("p1")).toContain("/api/plaza/posts/p1/download");
  expect(api.plazaReferenceDownloadUrl("p1")).toContain("/reference");
  expect(api.plazaPreviewUrl("p1")).toContain("/preview");
  expect(api.avatarUrl("u1")).toContain("/api/avatars/u1");
});
