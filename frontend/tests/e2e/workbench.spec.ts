import { expect, test, type Page, type Route } from "@playwright/test";

import { expectNoImplementationWording, navigateTo } from "./helpers";

type Account = { id: string; username: string; role: "user" | "admin"; status: "active" };

async function fulfill(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function installWorkspaceRoutes(page: Page, account: Account): Promise<void> {
  const readiness = { status: "ready", checks: { gpu: { ok: true, message: "GPU cuda:0 可用" } } };
  const dashboard = { counts: { datasets: 1, voices: 1, jobs: 1, active_jobs: 0, syntheses: 1 }, recent_jobs: [], voices: [], readiness };
  const voices = { items: [{ id: "ready-1", display_name: "我的旁白", dataset_id: "dataset-1", status: "ready", created_at: "2026-09-06T00:00:00Z", can_synthesize: true }] };
  const jobs = { items: [{ id: "synthesis-1", kind: "synthesize", status: "queued", error_code: null, public_message: null, created_at: "2026-09-06T00:00:00Z", started_at: null, finished_at: null }] };
  const syntheses = { items: [{ job_id: "blocked-1", voice_profile_id: "ready-1", text_lang: "zh", status: "failed", download_ready: false, watermark_probability: null, fingerprint_anomaly: null, created_at: "2026-09-06T00:00:00Z" }] };
  await page.route("**/api/auth/me", (route) => fulfill(route, account));
  await page.route("**/api/health/ready", (route) => fulfill(route, readiness));
  await page.route("**/api/dashboard", (route) => fulfill(route, dashboard));
  await page.route("**/api/datasets", (route) => fulfill(route, { items: [] }));
  await page.route("**/api/voices", (route) => fulfill(route, voices));
  await page.route("**/api/jobs**", (route) => fulfill(route, jobs));
  await page.route("**/api/syntheses", (route) =>
    fulfill(route, route.request().method() === "POST" ? { job_id: "synthesis-1", status: "queued" } : syntheses, route.request().method() === "POST" ? 202 : 200),
  );
  await page.route("**/api/safety/detect-watermark", (route) => fulfill(route, { job_id: "blocked-1", probability: 0.01, payload: null, payload_matches_job: false }));
  await page.route("**/api/admin/users", (route) => fulfill(route, { items: [{ id: "user-2", username: "alice", role: "user", status: "active", created_at: "2026-09-06T00:00:00Z", dataset_count: 1, voice_count: 1, job_count: 1 }] }));
  await page.route("**/api/admin/audit-events**", (route) => fulfill(route, { items: [], next_before_id: null }));
}

test("normal user can queue a ready voice but cannot access admin or download a blocked result", async ({ page }) => {
  await installWorkspaceRoutes(page, { id: "user-1", username: "alice", role: "user", status: "active" });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "管理中心" })).toHaveCount(0);
  await navigateTo(page, "语音创作");
  await page.getByRole("combobox", { name: "选择音色" }).click();
  await page.getByRole("option", { name: "我的旁白" }).click();
  await page.getByLabel("合成文本").fill("你好，世界");
  await page.getByLabel("我确认已获得声音授权").check();
  await page.getByRole("button", { name: "开始安全合成" }).click();
  await expect(
    page.getByRole("region", { name: "合成结果" }).getByText(/正在生成并完成质量检查/),
  ).toBeVisible();
  await navigateTo(page, "安全中心");
  await expect(page.getByRole("list", { name: "安全验证流程" })).toBeVisible();
  await expect(page.getByRole("button", { name: "复检来源" })).toBeVisible();
  await expect(page.getByRole("button", { name: "下载音频" })).not.toBeVisible();
  await page.getByRole("button", { name: "复检来源" }).click();
  await expect(page.getByText(/无法确认来源/).first()).toBeVisible();
  await expectNoImplementationWording(page);
});

test("admin sees the management entry with the user table", async ({ page }) => {
  await installWorkspaceRoutes(page, { id: "admin-1", username: "admin", role: "admin", status: "active" });
  await page.goto("/");
  await navigateTo(page, "管理中心");
  await expect(page.getByRole("table", { name: "用户列表" })).toBeVisible();
  await expect(page.getByText("alice", { exact: true })).toBeVisible();
  await expectNoImplementationWording(page);
});
