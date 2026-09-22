import { expect, test } from "@playwright/test";

import { expectNoImplementationWording, navigateTo } from "./helpers";

test("local user queues a verified synthesis from a ready voice", async ({ page }) => {
  const account = { id: "user-1", username: "alice", role: "user", status: "active" };
  const dashboard = {
    counts: { datasets: 0, voices: 1, jobs: 0, active_jobs: 0, syntheses: 0 },
    recent_jobs: [],
    voices: [],
    readiness: { status: "not_ready", checks: {} },
  };
  const voices = {
    items: [
      { id: "profile-1", display_name: "我的旁白", dataset_id: "dataset-1", status: "ready", created_at: "2026-09-06T00:00:00Z", can_synthesize: true },
      { id: "training-1", display_name: "训练中音色", dataset_id: "dataset-2", status: "training", created_at: "2026-09-06T00:00:00Z", can_synthesize: false },
    ],
  };
  const jobs = {
    items: [
      { id: "job-1", kind: "synthesize", status: "succeeded", error_code: null, public_message: null, created_at: "2026-09-06T00:00:00Z", started_at: null, finished_at: "2026-09-06T00:00:01Z" },
    ],
  };
  const syntheses = {
    items: [
      { job_id: "job-1", voice_profile_id: "profile-1", text_lang: "zh", status: "succeeded", download_ready: true, watermark_probability: 0.97, fingerprint_anomaly: false, speaker_similarity: 0.96, quality_warning_codes: [], created_at: "2026-09-06T00:00:00Z" },
    ],
  };

  await page.route("**/api/auth/me", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(account) });
  });
  await page.route("**/api/health/ready", async (route) => {
    await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ status: "not_ready", checks: {} }) });
  });
  await page.route("**/api/dashboard", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(dashboard) });
  });
  await page.route("**/api/voices", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(voices) });
  });
  await page.route("**/api/jobs**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(jobs) });
  });
  await page.route("**/api/syntheses", async (route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({ job_id: "job-1", status: "queued" }) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(syntheses) });
  });

  await page.goto("/");
  await navigateTo(page, "语音创作");
  await page.getByRole("combobox", { name: "选择音色" }).click();
  await page.getByRole("option", { name: "我的旁白" }).click();
  await expect(page.getByRole("option", { name: "training-1" })).toHaveCount(0);
  await page.getByLabel("合成文本").fill("你好，世界");
  await expect(page.getByRole("button", { name: "开始安全合成" })).toBeDisabled();
  await page.getByLabel("我确认已获得声音授权").check();
  await expect(page.getByRole("button", { name: "开始安全合成" })).toBeEnabled();
  await page.getByRole("button", { name: "开始安全合成" }).click();

  await expect(page.getByRole("button", { name: "下载音频" })).toBeVisible();
  await expectNoImplementationWording(page);
});
