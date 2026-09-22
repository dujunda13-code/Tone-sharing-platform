import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import tokensCss from "../styles/tokens.css?raw";
import componentsCss from "../styles/components.css?raw";
import { AudioResultCard } from "../components/AudioResultCard";
import { AtmosphericPanel } from "../components/AtmosphericPanel";
import { Disclosure } from "../components/Disclosure";
import { PasswordField } from "../components/PasswordField";

describe("ElevenLabs 设计基元", () => {
  afterEach(() => cleanup());

  it("keeps password visibility reusable after blur", async () => {
    const actor = userEvent.setup();
    render(
      <PasswordField
        id="password"
        label="密码"
        value="secret88"
        onChange={() => {}}
        autoComplete="current-password"
      />,
    );
    const toggle = screen.getByRole("button", { name: "显示密码" });
    await actor.click(toggle);
    expect(screen.getByLabelText("密码")).toHaveAttribute("type", "text");
    // Move focus away, then toggle back — the control must stay usable.
    await actor.tab();
    await actor.click(screen.getByRole("button", { name: "隐藏密码" }));
    expect(screen.getByLabelText("密码")).toHaveAttribute("type", "password");
    await actor.click(screen.getByRole("button", { name: "显示密码" }));
    expect(screen.getByLabelText("密码")).toHaveAttribute("type", "text");
  });

  it("suppresses native password reveal widgets beside the custom toggle", () => {
    expect(componentsCss).toMatch(
      /\.password-field input::-ms-reveal,\s*\.password-field input::-ms-clear\s*\{\s*display:\s*none;/,
    );
    expect(componentsCss).toMatch(
      /\.password-field input::-webkit-credentials-auto-fill-button\s*\{\s*visibility:\s*hidden;/,
    );
  });

  it("keeps technical details collapsed initially and expands on demand", async () => {
    const actor = userEvent.setup();
    render(
      <Disclosure label="技术详情">
        <span>voice-123</span>
      </Disclosure>,
    );
    expect(screen.queryByText("voice-123")).toBeNull();
    await actor.click(screen.getByRole("button", { name: "技术详情" }));
    expect(screen.getByText("voice-123")).toBeVisible();
    await actor.click(screen.getByRole("button", { name: "技术详情" }));
    expect(screen.queryByText("voice-123")).toBeNull();
  });

  it("offers playback and download only for ready audio with a quality warning", () => {
    render(
      <AudioResultCard
        src="/api/syntheses/job-1/audio"
        downloadName="合成_job-1.wav"
        similarity={0.74}
        warningCodes={["SPEAKER_SIMILARITY_BELOW_RECOMMENDED"]}
      />,
    );
    expect(screen.getByRole("button", { name: "下载音频" })).toHaveAttribute(
      "download",
      "合成_job-1.wav",
    );
    expect(screen.getByText(/相似度 74\.0%，低于建议值 90%/)).toBeInTheDocument();
    expect(document.querySelector("audio")).not.toBeNull();
  });

  it("defines the approved editorial tokens", () => {
    const css = tokensCss;
    expect(css).toContain("--canvas: #f5f5f5");
    expect(css).toContain("--ink: #0c0a09");
    expect(css).toContain("--card: #ffffff");
    expect(css).toContain("--hairline: #e7e5e4");
    expect(css).toContain("--radius-pill: 9999px");
    expect(css).toContain("--radius-card: 16px");
    expect(css).toContain("--radius-atmosphere: 24px");
    expect(css).toContain("--content-max: 1200px");
    expect(css).toContain('--font-display: "EB Garamond", "Times New Roman", serif');
    expect(css).toContain('--font-body: "Inter", sans-serif');
  });

  it("declares the five atmospheric gradient tones and uses them only as atmosphere", () => {
    const css = tokensCss;
    for (const tone of ["mint", "peach", "lavender", "sky", "rose"]) {
      expect(css).toContain(`--gradient-${tone}:`);
    }
    render(
      <AtmosphericPanel tone="mint">
        <h1>让每一种声音，都能被清晰表达。</h1>
      </AtmosphericPanel>,
    );
    const panel = screen.getByText(/让每一种声音/).closest(".atmospheric-panel");
    expect(panel).not.toBeNull();
    expect(panel).toHaveClass("tone-mint");
  });
});
