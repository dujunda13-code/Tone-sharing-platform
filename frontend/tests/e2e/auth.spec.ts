import { expect, test } from "@playwright/test";

import { assertResponsiveNav, expectNoImplementationWording } from "./helpers";

test("local registration and logout keep the workspace behind auth", async ({ page }) => {
  let authenticated = false;
  const account = { id: "user-1", username: "alice", role: "user", status: "active" };
  const dashboard = {
    counts: { datasets: 0, voices: 0, jobs: 0, active_jobs: 0, syntheses: 0 },
    recent_jobs: [],
    voices: [],
    readiness: { status: "not_ready", checks: {} },
  };

  await page.route("**/api/auth/me", async (route) => {
    if (authenticated) {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(account) });
      return;
    }
    await route.fulfill({
      status: 401,
      contentType: "application/json",
      body: JSON.stringify({ error: { code: "AUTH_REQUIRED", message: "需要登录", details: {}, request_id: "" } }),
    });
  });
  await page.route("**/api/auth/register", async (route) => {
    await route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify(account) });
  });
  await page.route("**/api/auth/login", async (route) => {
    authenticated = true;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(account) });
  });
  await page.route("**/api/auth/logout", async (route) => {
    authenticated = false;
    await route.fulfill({ status: 204, body: "" });
  });
  await page.route("**/api/health/ready", async (route) => {
    await route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ status: "not_ready", checks: {} }),
    });
  });
  await page.route("**/api/dashboard", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(dashboard) });
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "欢迎回来" })).toBeVisible();
  await expect(page.getByRole("button", { name: "工作台" })).not.toBeVisible();
  await expectNoImplementationWording(page);

  await page.getByRole("button", { name: "立即注册" }).click();
  await page.getByLabel("用户名").fill("alice");
  await page.getByLabel("密码", { exact: true }).fill("Correct-Horse-42");
  await page.getByLabel("确认密码", { exact: true }).fill("Correct-Horse-42");
  await page.getByRole("button", { name: "注册" }).click();
  await expect(page.getByText("注册成功，请使用新账号登录。")).toBeVisible();

  await page.getByLabel("用户名").fill("alice");
  await page.getByLabel("密码", { exact: true }).fill("Correct-Horse-42");
  await page.getByRole("button", { name: "登录" }).click();
  await expect(page.getByText(/让每一种声音/)).toBeVisible();
  await assertResponsiveNav(page);
  await page.getByRole("button", { name: "退出登录" }).click();
  await expect(page.getByRole("heading", { name: "欢迎回来" })).toBeVisible();
});
