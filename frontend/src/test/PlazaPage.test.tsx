import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import { api } from "../api/client";
import { PlazaPage } from "../pages/PlazaPage";
import { installWorkspaceApiMock, ZERO_SHOT_VOICE } from "./workspaceApiMock";

const POST = {
  id: "post-1",
  voice: {
    voice_profile_id: "voice-1",
    display_name: "零样本演示音色",
    status: "ready",
    reference_count: 2,
    reference_emotions: ["happy", "neutral"],
  },
  author: { user_id: "user-2", display_name: "bob", has_avatar: false },
  description: "温柔的旁白音色",
  like_count: 3,
  comment_count: 0,
  favorite_count: 1,
  liked_by_me: false,
  favorited_by_me: false,
  created_at: "2026-09-17T00:00:00Z",
};

afterEach(() => {
  vi.unstubAllGlobals();
  cleanup();
});

test("renders plaza posts with voice name, author, counts and download links", async () => {
  installWorkspaceApiMock({ plazaPosts: [POST] });
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  expect(await screen.findByText("零样本演示音色")).toBeInTheDocument();
  expect(screen.getByText("零样本演示音色").closest("article")).toHaveClass("plaza-directory-row");
  expect(screen.getByText("bob")).toBeInTheDocument();
  expect(screen.getByText("bob").closest(".plaza-post-identity")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "点赞" })).toHaveTextContent("3");
  expect(screen.getByRole("button", { name: "收藏" })).toHaveTextContent("1");
  const downloadZip = screen.getByRole("link", { name: "下载音色包" });
  expect(downloadZip).toHaveAttribute("href", api.plazaDownloadUrl("post-1"));
  expect(screen.getByRole("link", { name: "主参考音频" })).toHaveAttribute(
    "href",
    api.plazaReferenceDownloadUrl("post-1"),
  );
});

test("lists published voices in one continuous accessible directory instead of post cards", async () => {
  installWorkspaceApiMock({ plazaPosts: [POST] });
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  const directory = await screen.findByRole("list", { name: "已发布音色" });
  const [row] = within(directory).getAllByRole("listitem");

  expect(row).not.toHaveClass("plaza-card");
  expect(within(row).getByText("零样本演示音色")).toBeInTheDocument();
});

test("renders an editorial hero while keeping the accessible plaza controls", async () => {
  installWorkspaceApiMock({ plazaPosts: [POST] });
  const { container } = render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  expect(await screen.findByText("让声音，被更多人听见")).toBeInTheDocument();
  expect(container.querySelector(".plaza-hero.atmospheric-panel .plaza-hero-waves")).toBeInTheDocument();
  expect(screen.getByRole("tablist", { name: "广场过滤" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "发布音色" })).toBeInTheDocument();
});

test("renders a real empty plaza state with publish and create actions", async () => {
  installWorkspaceApiMock({ plazaPosts: [] });
  const onNavigate = vi.fn();
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={onNavigate} onSynthesize={vi.fn()} />);

  expect(await screen.findByText("还没有公开音色")).toBeInTheDocument();
  expect(screen.queryByText("零样本演示音色")).not.toBeInTheDocument();
  await user.click(within(screen.getByRole("region", { name: "广场空状态" })).getByRole("button", { name: "去创建音色" }));
  expect(onNavigate).toHaveBeenCalledWith("create");
});

test("filter tabs switch list queries", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [POST] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("tab", { name: "我的发布" }));
  const calls = (fetchMock as unknown as { mock: { calls: unknown[][] } }).mock.calls;
  expect(
    calls.some(([url]) => String(url).includes("/api/plaza/posts?") && String(url).includes("mine=true")),
  ).toBe(true);
});

test("立即创作 forwards voice id and label", async () => {
  installWorkspaceApiMock({ plazaPosts: [POST] });
  const onSynthesize = vi.fn();
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={onSynthesize} />);

  await user.click(await screen.findByRole("button", { name: "立即创作" }));

  expect(onSynthesize).toHaveBeenCalledWith("voice-1", "零样本演示音色");
});

const INTERACTIVE_POST = {
  ...POST,
  id: "post-2",
  liked_by_me: false,
  favorited_by_me: false,
};

test("like button posts then reloads the plaza list", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [INTERACTIVE_POST] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("button", { name: "点赞" }));

  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(
    calls.some(([url, init]) => String(url).includes("/api/plaza/posts/post-2/likes") && init?.method === "POST"),
  ).toBe(true);
});

test("like action stays selected locally so the next click cancels it", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [INTERACTIVE_POST] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("button", { name: "点赞" }));
  expect(await screen.findByRole("button", { name: "取消点赞" })).toHaveAttribute("aria-pressed", "true");

  await user.click(screen.getByRole("button", { name: "取消点赞" }));
  await waitFor(() => {
    const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
    expect(calls.some(([url, init]) => String(url).includes("/api/plaza/posts/post-2/likes") && init?.method === "DELETE")).toBe(true);
  });
  expect(await screen.findByRole("button", { name: "点赞" })).toBeInTheDocument();
});

test("favorite action stays selected locally so the next click cancels it", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [INTERACTIVE_POST] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("button", { name: "收藏" }));
  expect(await screen.findByRole("button", { name: "取消收藏" })).toHaveAttribute("aria-pressed", "true");

  await user.click(screen.getByRole("button", { name: "取消收藏" }));
  await waitFor(() => {
    const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
    expect(calls.some(([url, init]) => String(url).includes("/api/plaza/posts/post-2/favorites") && init?.method === "DELETE")).toBe(true);
  });
  expect(await screen.findByRole("button", { name: "收藏" })).toBeInTheDocument();
});

test("removes a voice from my favorites after its cancellation succeeds", async () => {
  const favoritedPost = { ...INTERACTIVE_POST, favorited_by_me: true };
  installWorkspaceApiMock({ plazaPosts: [favoritedPost] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("tab", { name: "我的收藏" }));
  await user.click(await screen.findByRole("button", { name: "取消收藏" }));

  await waitFor(() => expect(screen.queryByText("零样本演示音色")).not.toBeInTheDocument());
});

test("favorite toggle and comment flow use plaza APIs", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [INTERACTIVE_POST], plazaComments: [] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("button", { name: "收藏" }));
  await user.click(await screen.findByRole("button", { name: /^评论/ }));

  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(calls.some(([url, init]) => String(url).includes("/favorites") && init?.method === "POST")).toBe(true);
  expect(calls.some(([url]) => String(url).includes("/api/plaza/posts/post-2/comments"))).toBe(true);

  await user.type(await screen.findByRole("textbox", { name: "发表评论" }), "非常好听");
  await user.click(screen.getByRole("button", { name: "发表评论" }));
  expect(calls.some(([url, init]) => String(url).includes("/comments") && init?.method === "POST")).toBe(true);
});

test("import modal requires authorization confirmation", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [INTERACTIVE_POST] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("button", { name: "导入到我的音色" }));
  const confirm = screen.getByRole("button", { name: "确认导入" });
  expect(confirm).toBeDisabled();

  await user.click(screen.getByRole("checkbox", { name: /已获得该音色的使用授权/ }));
  expect(confirm).toBeEnabled();
  await user.click(confirm);

  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(calls.some(([url, init]) => String(url).includes("/import") && init?.method === "POST")).toBe(true);
});

test("publish modal lists own ready voices excluding published voices", async () => {
  const fetchMock = installWorkspaceApiMock({
    plazaPosts: [INTERACTIVE_POST],
    voices: [{ ...ZERO_SHOT_VOICE }, { ...ZERO_SHOT_VOICE, id: "voice-2", display_name: "已发布音色" }],
  });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("button", { name: "发布音色" }));

  expect(screen.getByRole("dialog", { name: "发布音色" })).toHaveClass("plaza-modal-content");

  const dialog = screen.getByRole("dialog", { name: "发布音色" });
  const picker = await within(dialog).findByRole("combobox", { name: "发布音色选择" });
  expect(picker).toHaveTextContent("零样本演示音色");
  await user.click(within(dialog).getByRole("button", { name: "发布音色" }));

  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(calls.some(([url, init]) => String(url).includes("/api/plaza/posts") && init?.method === "POST")).toBe(true);
});

test("preview requests the authenticated preview endpoint", async () => {
  const fetchMock = installWorkspaceApiMock({ plazaPosts: [INTERACTIVE_POST] });
  const user = userEvent.setup();
  render(<PlazaPage onNavigate={vi.fn()} onSynthesize={vi.fn()} />);

  await user.click(await screen.findByRole("button", { name: "试听" }));

  const calls = (fetchMock as unknown as { mock: { calls: Array<[unknown, RequestInit | undefined]> } }).mock.calls;
  expect(calls.some(([url]) => String(url).includes("/preview"))).toBe(true);
});
