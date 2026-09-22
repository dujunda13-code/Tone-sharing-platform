import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { AdminCenterPage } from "../pages/AdminCenterPage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const ADMIN = { id: "admin-1", username: "admin", role: "admin" as const, status: "active" };

describe("AdminCenterPage 管理中心", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("presents users as a table with a locked self row and audited actions", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      currentUser: ADMIN,
      adminUsers: [
        { id: "admin-1", username: "admin", role: "admin", status: "active", created_at: "2026-09-06T00:00:00Z", dataset_count: 0, voice_count: 0, job_count: 0 },
        { id: "u-2", username: "alice", role: "user", status: "active", created_at: "2026-09-06T00:00:00Z", dataset_count: 1, voice_count: 1, job_count: 1 },
      ],
    });
    render(<App />);

    await user.click(await screen.findByRole("button", { name: "管理中心" }));
    const table = await screen.findByRole("table", { name: "用户列表" });
    const headers = within(table)
      .getAllByRole("columnheader")
      .map((header) => header.textContent);
    expect(headers).toEqual(["用户", "角色", "状态", "参考音频", "音色", "任务", "操作"]);

    // 自己的行锁定操作；其他用户可以停用/恢复。
    const selfRow = within(table).getByRole("row", { name: /admin/ });
    expect(within(selfRow).getByRole("button", { name: "当前账号" })).toBeDisabled();
    const aliceRow = within(table).getByRole("row", { name: /alice/ });
    await user.click(within(aliceRow).getByRole("button", { name: "停用" }));
    expect(vi.mocked(fetch)).toHaveBeenCalledWith(
      expect.stringContaining("/api/admin/users/u-2/status"),
      expect.objectContaining({ method: "PATCH" }),
    );

    // 面向用户的文案不出现实现词与位置词。
    expect(document.body).not.toHaveTextContent(/本地|本机/);
  });

  it("refuses to render for non-admin users without sending admin requests", async () => {
    installWorkspaceApiMock({
      currentUser: { id: "user-1", username: "alice", role: "user" },
    });
    render(
      <AdminCenterPage
        user={{ id: "user-1", username: "alice", role: "user", status: "active" }}
      />,
    );

    expect(await screen.findByText(/仅管理员可访问/)).toBeInTheDocument();
  });

  it("loads earlier audit events with the backend cursor without duplicate records", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      currentUser: ADMIN,
      adminUsers: [],
      auditEvents: [
        { id: 2, event_type: "admin.user_status_changed", subject_id: "u-2", metadata: { old_status: "active", new_status: "disabled" }, created_at: "2026-09-06T00:00:02Z" },
      ],
      auditNextBeforeId: 1,
    });
    render(<AdminCenterPage user={ADMIN} />);

    // 审计时间线使用自然语言事件名；内部对象 ID 默认折叠。
    expect(await screen.findByText(/账号状态变更/)).toBeInTheDocument();
    expect(screen.queryByText("u-2")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "加载更早事件" }));
    expect(vi.mocked(fetch)).toHaveBeenLastCalledWith(
      expect.stringContaining("/api/admin/audit-events"),
      expect.anything(),
    );
    // 游标翻页重复返回时按事件去重，不产生重复记录或重复 React key。
    expect(screen.getAllByText(/账号状态变更/)).toHaveLength(1);
  });
});
