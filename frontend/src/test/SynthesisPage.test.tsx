import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SynthesisPage } from "../pages/SynthesisPage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

describe("SynthesisPage（委托语音创作工作台）", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("requires a ready voice, text, and consent before safe synthesis", async () => {
    const user = userEvent.setup();
    installWorkspaceApiMock({
      voices: [
        { id: "ready-1", display_name: "温柔旁白", dataset_id: "d1", status: "ready", created_at: "2026-09-06T00:00:00Z", can_synthesize: true },
      ],
    });
    render(<SynthesisPage />);

    await user.click(await screen.findByRole("combobox", { name: "选择音色" }));
    await user.click(screen.getByRole("option", { name: "温柔旁白" }));
    await user.type(screen.getByLabelText("合成文本"), "你好，世界");
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeDisabled();
    await user.click(screen.getByLabelText("我确认已获得声音授权"));
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeEnabled();
  });

  it("excludes voices that are not marked synthesizable", async () => {
    installWorkspaceApiMock({
      voices: [
        { id: "training-1", dataset_id: "d2", status: "training", created_at: "2026-09-06T00:00:00Z", can_synthesize: false },
      ],
    });
    render(<SynthesisPage />);

    expect(await screen.findByText(/暂无可用音色/)).toBeInTheDocument();
    expect(screen.getByLabelText("合成文本")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始安全合成" })).toBeDisabled();
    expect(vi.mocked(fetch)).toHaveBeenCalled();
  });
});
