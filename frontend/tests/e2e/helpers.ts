import { expect, type Locator, type Page } from "@playwright/test";

/** E2E 公共助手：导航在桌面为顶部主导航，窄屏折叠进菜单按钮。 */

export function isMobileViewport(page: Page): boolean {
  return (page.viewportSize()?.width ?? 1280) < 768;
}

export function desktopNav(page: Page): Locator {
  return page.getByRole("navigation", { name: "主导航" });
}

export async function navigateTo(page: Page, label: string): Promise<void> {
  if (isMobileViewport(page)) {
    await page.getByRole("button", { name: "打开导航菜单" }).click();
    await page
      .getByRole("navigation", { name: "移动端导航" })
      .getByRole("button", { name: label })
      .click();
    return;
  }
  await desktopNav(page).getByRole("button", { name: label }).click();
}

/** 每个视口的导航契约：桌面主导航常驻；移动端通过菜单按钮可达。 */
export async function assertResponsiveNav(page: Page): Promise<void> {
  if (isMobileViewport(page)) {
    const toggle = page.getByRole("button", { name: "打开导航菜单" });
    await expect(toggle).toBeVisible();
    await toggle.click();
    await expect(page.getByRole("navigation", { name: "移动端导航" })).toBeVisible();
    await toggle.click();
    return;
  }
  await expect(desktopNav(page)).toBeVisible();
}

/** 全局实现词契约：正文不出现"本地/本机"，页面不横向溢出。 */
export async function expectNoImplementationWording(page: Page): Promise<void> {
  await expect(page.locator("body")).not.toContainText(/本地|本机/);
  await expect(page.locator("body")).toHaveCSS("overflow-x", "hidden");
}
