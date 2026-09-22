import { expect, test, type Page, type Route } from "@playwright/test";

import { assertResponsiveNav, expectNoImplementationWording, navigateTo } from "./helpers";

type Account = { id: string; username: string; role: "user" | "admin"; status: "active" };

async function fulfill(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function installVisualRoutes(page: Page, account: Account | null): Promise<void> {
  const readiness = { status: "ready", checks: {} };
  const dashboard = {
    counts: { datasets: 1, voices: 2, jobs: 3, active_jobs: 1, syntheses: 2 },
    recent_jobs: [
      {
        id: "job-2",
        kind: "synthesize",
        status: "succeeded",
        error_code: null,
        public_message: "合成完成，结果已通过全部安全检查。",
        created_at: "2026-09-13T10:00:00Z",
        started_at: null,
        finished_at: "2026-09-13T10:02:00Z",
      },
    ],
    voices: [
      {
        id: "profile-1",
        display_name: "我的旁白",
        dataset_id: "dataset-1",
        status: "ready",
        mode: "zero_shot",
        base_model_id: "gpt-sovits-v2proplus-official",
        reference_emotions: ["happy"],
        reference_snr_db: null,
        quality_warning_codes: [],
        can_synthesize: true,
        created_at: "2026-09-12T00:00:00Z",
      },
    ],
    readiness,
  };
  const voices = {
    items: [
      { id: "profile-1", display_name: "我的旁白", dataset_id: "dataset-1", status: "ready", mode: "zero_shot", base_model_id: "gpt-sovits-v2proplus-official", reference_emotions: ["happy"], reference_snr_db: null, quality_warning_codes: [], can_synthesize: true, created_at: "2026-09-12T00:00:00Z" },
      { id: "profile-2", display_name: "温柔旁白", dataset_id: "dataset-2", status: "ready", mode: "zero_shot", base_model_id: "gpt-sovits-v2proplus-official", reference_emotions: ["calm"], reference_snr_db: 18.4, quality_warning_codes: [], can_synthesize: true, created_at: "2026-09-11T00:00:00Z" },
    ],
  };
  const jobs = {
    items: [
      { id: "job-1", kind: "synthesize", status: "succeeded", error_code: null, public_message: "合成完成，结果已通过全部安全检查。", created_at: "2026-09-13T10:00:00Z", started_at: "2026-09-13T10:00:01Z", finished_at: "2026-09-13T10:02:00Z" },
      { id: "job-2", kind: "preprocess", status: "failed", error_code: "AUDIO_TOO_SHORT", public_message: "有效语音时长不足", created_at: "2026-09-12T10:00:00Z", started_at: "2026-09-12T10:00:01Z", finished_at: "2026-09-12T10:00:20Z" },
    ],
  };
  const syntheses = {
    items: [
      { job_id: "job-1", voice_profile_id: "profile-1", text_lang: "zh", status: "succeeded", download_ready: true, watermark_probability: 0.99, fingerprint_anomaly: false, speaker_similarity: 0.762, quality_warning_codes: ["SPEAKER_SIMILARITY_BELOW_RECOMMENDED"], created_at: "2026-09-13T10:00:00Z" },
      { job_id: "job-3", voice_profile_id: "profile-2", text_lang: "zh", status: "failed", download_ready: false, watermark_probability: null, fingerprint_anomaly: null, speaker_similarity: null, quality_warning_codes: [], created_at: "2026-09-12T10:00:00Z" },
    ],
  };
  const adminUsers = {
    items: [
      { id: "admin-1", username: "admin", role: "admin", status: "active", created_at: "2026-09-01T00:00:00Z", dataset_count: 2, voice_count: 3, job_count: 9 },
      { id: "user-1", username: "alice", role: "user", status: "active", created_at: "2026-09-06T00:00:00Z", dataset_count: 1, voice_count: 2, job_count: 3 },
    ],
  };
  const auditEvents = {
    items: [
      { id: 3, event_type: "admin.user_status_changed", subject_id: "user-2", metadata: { old_status: "active", new_status: "disabled" }, created_at: "2026-09-13T12:00:00Z" },
      { id: 2, event_type: "synthesis.published", subject_id: "job-1", metadata: {}, created_at: "2026-09-13T10:02:00Z" },
    ],
    next_before_id: null,
  };
  const profile = account
    ? {
        user_id: account.id,
        username: account.username,
        display_name: account.username === "alice" ? "小艾" : account.username,
        bio: "让声音被更多人听见。",
        has_avatar: false,
        stats: { published: 0, likes_received: 0 },
      }
    : null;

  await page.route("**/api/auth/me", (route) =>
    account
      ? fulfill(route, account)
      : fulfill(route, { error: { code: "AUTH_REQUIRED", message: "需要登录", details: {}, request_id: "" } }, 401),
  );
  await page.route("**/api/health/ready", (route) => fulfill(route, readiness));
  await page.route("**/api/dashboard", (route) => fulfill(route, dashboard));
  await page.route("**/api/datasets", (route) => fulfill(route, { items: [] }));
  await page.route("**/api/voices", (route) => fulfill(route, voices));
  await page.route("**/api/jobs**", (route) => fulfill(route, jobs));
  await page.route("**/api/syntheses", (route) =>
    fulfill(route, syntheses, route.request().method() === "POST" ? 202 : 200),
  );
  await page.route("**/api/safety/detect-watermark", (route) =>
    fulfill(route, { job_id: "job-1", probability: 0.99, payload: 1, payload_matches_job: true }),
  );
  await page.route("**/api/admin/users", (route) => fulfill(route, adminUsers));
  await page.route("**/api/admin/audit-events**", (route) => fulfill(route, auditEvents));
  await page.route("**/api/plaza/posts**", (route) => fulfill(route, { items: [], total: 0 }));
  await page.route("**/api/plaza/comments", (route) => fulfill(route, { items: [], total: 0 }));
  await page.route("**/api/notifications**", (route) => fulfill(route, { items: [], total: 0 }));
  await page.route("**/api/notifications/unread-count", (route) => fulfill(route, { count: 0 }));
  await page.route("**/api/users/me/profile", (route) =>
    profile
      ? fulfill(route, profile)
      : fulfill(route, { error: { code: "AUTH_REQUIRED", message: "需要登录", details: {}, request_id: "" } }, 401),
  );
}

async function capture(page: Page, name: string, project: string): Promise<void> {
  await expectNoImplementationWording(page);
  await page.screenshot({
    path: `test-results/visual/${name}-${project}.png`,
    fullPage: true,
    animations: "disabled",
  });
}

test("captures the login page at the active viewport", async ({ page }, testInfo) => {
  await installVisualRoutes(page, null);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "欢迎回来" })).toBeVisible();
  await capture(page, "login", testInfo.project.name);
});

test("aligns the login registration link with the form fields", async ({ page }) => {
  await installVisualRoutes(page, null);
  await page.goto("/");
  await expect(page.getByRole("button", { name: "立即注册" })).toBeVisible();

  const metrics = await page.evaluate(() => {
    const registerPrompt = document.querySelector<HTMLElement>(".auth-register-action .hint");
    const registerLink = document.querySelector<HTMLElement>(".auth-register-link");
    return {
      promptCenterY: registerPrompt ? (registerPrompt.getBoundingClientRect().top + registerPrompt.getBoundingClientRect().bottom) / 2 : 0,
      linkCenterY: registerLink ? (registerLink.getBoundingClientRect().top + registerLink.getBoundingClientRect().bottom) / 2 : 0,
      promptRight: registerPrompt?.getBoundingClientRect().right ?? 0,
      linkLeft: registerLink?.getBoundingClientRect().left ?? 0,
    };
  });

  expect(metrics.promptCenterY).toBeGreaterThan(0);
  expect(Math.abs(metrics.promptCenterY - metrics.linkCenterY)).toBeLessThan(2);
  expect(metrics.linkLeft).toBeGreaterThan(metrics.promptRight);
});

test("captures the registration page at the active viewport", async ({ page }, testInfo) => {
  await installVisualRoutes(page, null);
  await page.goto("/");
  await page.getByRole("button", { name: "立即注册" }).click();
  await expect(page.getByRole("heading", { name: "创建账号", exact: true })).toBeVisible();
  await capture(page, "register", testInfo.project.name);
});

test("keeps the authentication composition inside a narrow viewport", async ({ page }) => {
  await installVisualRoutes(page, null);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "欢迎回来" })).toBeVisible();
  const metrics = await page.evaluate(() => {
    const art = document.querySelector<HTMLElement>(".auth-art-panel");
    const card = document.querySelector<HTMLElement>(".auth-card");
    return {
      viewportWidth: window.innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      artWidth: art?.getBoundingClientRect().width ?? 0,
      cardWidth: card?.getBoundingClientRect().width ?? 0,
    };
  });
  expect(metrics.documentWidth).toBeLessThanOrEqual(metrics.viewportWidth);
  expect(metrics.artWidth).toBeGreaterThan(0);
  expect(metrics.cardWidth).toBeGreaterThan(0);
});

test("captures every user page at the active viewport", async ({ page }, testInfo) => {
  await installVisualRoutes(page, { id: "user-1", username: "alice", role: "user", status: "active" });
  await page.goto("/");
  await expect(page.getByText(/让每一种声音/)).toBeVisible();
  await assertResponsiveNav(page);

  const stops = [
    { label: "我的音色", marker: /我的旁白/, shot: "voices" },
    { label: "创建音色", marker: /快速克隆音色/, shot: "create-voice" },
    { label: "语音创作", marker: /开始安全合成/, shot: "synthesis" },
    { label: "任务中心", marker: /音频分析和语音合成任务/, shot: "tasks" },
    { label: "安全中心", marker: /嵌入来源水印/, shot: "safety" },
  ] as const;
  for (const stop of stops) {
    await navigateTo(page, stop.label);
    await expect(page.getByText(stop.marker).first()).toBeVisible();
    await capture(page, stop.shot, testInfo.project.name);
  }
  await navigateTo(page, "工作台");
  await capture(page, "dashboard", testInfo.project.name);
});

test("captures the admin page at the active viewport", async ({ page }, testInfo) => {
  await installVisualRoutes(page, { id: "admin-1", username: "admin", role: "admin", status: "active" });
  await page.goto("/");
  await navigateTo(page, "管理中心");
  await expect(page.getByRole("table", { name: "用户列表" })).toBeVisible();
  await capture(page, "admin", testInfo.project.name);
});

test("captures the plaza empty state and personal center", async ({ page }, testInfo) => {
  await installVisualRoutes(page, { id: "user-1", username: "alice", role: "user", status: "active" });
  await page.goto("/");

  await navigateTo(page, "音色广场");
  await expect(page.locator(".plaza-empty-state")).toBeVisible();
  await capture(page, "plaza-empty", testInfo.project.name);

  await page.getByRole("button", { name: "打开个人中心" }).click();
  await expect(page.locator(".profile-hero")).toBeVisible();
  await expect(page.getByText("我的空间")).toHaveCount(0);
  await capture(page, "profile", testInfo.project.name);
});

test("keeps plaza and profile controls pill-shaped after interaction", async ({ page }) => {
  await installVisualRoutes(page, { id: "user-1", username: "alice", role: "user", status: "active" });
  await page.goto("/");

  await navigateTo(page, "音色广场");
  await page.getByRole("tab", { name: "我的发布" }).click();
  await page.getByLabel("搜索音色").focus();

  const plazaMetrics = await page.evaluate(() => {
    const value = (selector: string) => {
      const element = document.querySelector<HTMLElement>(selector);
      return element ? getComputedStyle(element).borderRadius : "";
    };
    const account = document.querySelector<HTMLElement>(".user-chip");
    const accountName = document.querySelector<HTMLElement>(".user-chip .user-name");
    return {
      activeFilterRadius: value('.plaza-filter-tabs button[aria-selected="true"]'),
      searchRadius: value("input.plaza-search"),
      sortRadius: value(".plaza-sort .dropdown-trigger"),
      accountBorderTopWidth: account ? getComputedStyle(account).borderTopWidth : "",
      accountNameDisplay: accountName ? getComputedStyle(accountName).display : "",
    };
  });

  expect(plazaMetrics.activeFilterRadius).toBe("9999px");
  expect(plazaMetrics.searchRadius).toBe("9999px");
  expect(plazaMetrics.sortRadius).toBe("9999px");
  expect(plazaMetrics.accountBorderTopWidth).toBe("0px");
  expect(plazaMetrics.accountNameDisplay).toBe("none");

  await page.getByRole("button", { name: "打开个人中心" }).click();
  await expect(page.locator(".profile-hero")).toBeVisible();
  await page.getByRole("tab", { name: "我的收藏" }).click();
  const profileTabRadius = await page.locator('.profile-tabs button[aria-selected="true"]').evaluate(
    (element) => getComputedStyle(element).borderRadius,
  );
  expect(profileTabRadius).toBe("9999px");
});

test("matches the nickname control to the profile bio control", async ({ page }) => {
  await installVisualRoutes(page, { id: "user-1", username: "alice", role: "user", status: "active" });
  await page.goto("/");
  await page.getByRole("button", { name: "打开个人中心" }).click();
  await expect(page.locator(".profile-hero")).toBeVisible();

  const metrics = await page.evaluate(() => {
    const nickname = document.querySelector<HTMLInputElement>('[aria-label="昵称"]');
    const values = (selector: string) => {
      const element = document.querySelector<HTMLElement>(selector);
      const style = element ? getComputedStyle(element) : null;
      return {
        borderRadius: style?.borderRadius ?? "",
        borderTopColor: style?.borderTopColor ?? "",
        paddingInlineStart: style?.paddingInlineStart ?? "",
      };
    };
    return {
      nicknameType: nickname?.getAttribute("type") ?? "",
      nickname: values('[aria-label="昵称"]'),
      bio: values(".profile-edit-panel textarea"),
    };
  });

  expect(metrics.nicknameType).toBe("text");
  expect(metrics.nickname).toEqual(metrics.bio);
});

test("shows styled plaza choices and a composed comment panel, and aligns voice columns", async ({ page }, testInfo) => {
  await installVisualRoutes(page, { id: "user-1", username: "alice", role: "user", status: "active" });
  const post = {
    id: "post-1",
    voice: { voice_profile_id: "profile-1", display_name: "温柔旁白", status: "ready", reference_count: 1, reference_emotions: ["happy"] },
    author: { user_id: "user-2", display_name: "小雅", has_avatar: false },
    description: "适合叙事与日常分享的温柔声音。",
    like_count: 3, comment_count: 1, favorite_count: 2,
    liked_by_me: false, favorited_by_me: false, created_at: "2026-09-17T00:00:00Z",
  };
  await page.route("**/api/plaza/posts**", (route) => fulfill(route, { items: [post], total: 1 }));
  await page.route("**/api/plaza/posts/post-1/comments", (route) => fulfill(route, { items: [{ id: "comment-1", post_id: "post-1", author: { user_id: "user-3", display_name: "小林", has_avatar: false }, content: "声音很自然", created_at: "2026-09-18T00:00:00Z" }], total: 1 }));
  await page.goto("/");
  await navigateTo(page, "音色广场");
  await expect(page.getByText("温柔旁白")).toBeVisible();
  await page.getByRole("combobox", { name: "排序方式" }).click();
  await expect(page.getByRole("listbox", { name: "排序方式" })).toBeVisible();
  await page.getByRole("option", { name: "最多点赞" }).click();
  await expect(page.getByRole("combobox", { name: "排序方式" })).toContainText("最多点赞");
  await page.getByRole("button", { name: /^评论/ }).click();
  await expect(page.getByRole("region", { name: "评论区" }).getByText("声音很自然")).toBeVisible();
  await capture(page, "plaza-comments", testInfo.project.name);

  await navigateTo(page, "我的音色");
  await expect(page.locator(".voice-library-rows .voice-row").first()).toBeVisible();
  if (page.viewportSize()!.width <= 640) {
    await expect(page.locator(".voice-library-rows .voice-row-state").first()).toBeVisible();
    await expect(page.locator(".voice-library-rows .voice-row-created").first()).toBeVisible();
  } else {
    const alignment = await page.evaluate(() => {
      const left = (selector: string) => document.querySelector(selector)?.getBoundingClientRect().left ?? 0;
      return [
        left(".voice-library-columns span:nth-child(2)") - left(".voice-library-rows .voice-row-tags"),
        left(".voice-library-columns span:nth-child(3)") - left(".voice-library-rows .voice-row-state"),
        left(".voice-library-columns span:nth-child(4)") - left(".voice-library-rows .voice-row-created"),
      ];
    });
    alignment.forEach((offset) => expect(Math.abs(offset)).toBeLessThan(2));
  }
});
