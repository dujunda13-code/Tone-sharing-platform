import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { AppShell } from "../components/AppShell";
import { CreateVoiceWizard } from "../components/CreateVoiceWizard";
import { AdminCenterPage } from "../pages/AdminCenterPage";
import { DashboardPage } from "../pages/DashboardPage";
import { SafetyCenterPage } from "../pages/SafetyCenterPage";
import { SynthesisStudioPage } from "../pages/SynthesisStudioPage";
import { TaskCenterPage } from "../pages/TaskCenterPage";
import { VoiceLibraryPage } from "../pages/VoiceLibraryPage";
import { installWorkspaceApiMock } from "./workspaceApiMock";

const USER = { id: "user-1", username: "林夕", role: "user" as const, status: "active" as const };

describe("预览图式视觉骨架", () => {
  afterEach(() => cleanup());

  it("uses a floating light app frame with icon controls and styled navigation", () => {
    const { container } = render(
      <AppShell
        user={USER}
        health={{ status: "ready", checks: {} }}
        activePage="dashboard"
        onNavigate={() => {}}
        onLogout={async () => {}}
      >
        <div>content</div>
      </AppShell>,
    );

    expect(container.querySelector(".preview-app-shell")).toBeInTheDocument();
    expect(container.querySelector(".preview-app-shell-fluid")).toBeInTheDocument();
    expect(container.querySelector(".preview-top-nav")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "主导航" })).toHaveClass("preview-nav");
    expect(screen.getByRole("button", { name: "搜索" })).toHaveClass("nav-icon-button");
    expect(screen.getByRole("button", { name: "通知" })).toHaveClass("nav-icon-button");
  });

  it("presents the reference audio upload as a branded dropzone", () => {
    installWorkspaceApiMock({ health: { status: "ready", checks: {} } });
    const { container } = render(<CreateVoiceWizard onNavigate={() => {}} />);

    expect(container.querySelector(".file-dropzone")).toBeInTheDocument();
    expect(screen.getByLabelText("授权音频文件")).toBeInTheDocument();
  });

  it("keeps the synthesis result workspace visible before the first job", async () => {
    installWorkspaceApiMock({
      voices: [
        {
          id: "voice-1",
          display_name: "我的旁白",
          dataset_id: "dataset-1",
          status: "ready",
          mode: "zero_shot",
          can_synthesize: true,
          reference_emotions: ["happy"],
          created_at: "2026-09-06T00:00:00Z",
        },
      ],
    });
    const { container } = render(<SynthesisStudioPage onNavigate={() => {}} />);

    expect(await screen.findByRole("region", { name: "合成结果" })).toBeInTheDocument();
    expect(container.querySelector(".result-placeholder")).toBeInTheDocument();
  });

  it("uses the approved board composition for every workspace destination", async () => {
    installWorkspaceApiMock({
      dashboard: {
        counts: { datasets: 2, voices: 2, jobs: 3, active_jobs: 1, syntheses: 4 },
        recent_jobs: [],
        voices: [],
        readiness: { status: "ready", checks: {} },
      },
      health: { status: "ready", checks: {} },
      voices: [
        {
          id: "voice-1",
          display_name: "我的旁白",
          dataset_id: "dataset-1",
          status: "ready",
          mode: "zero_shot",
          reference_emotions: ["happy"],
          can_synthesize: true,
          created_at: "2026-09-06T00:00:00Z",
        },
      ],
      jobs: [
        {
          id: "job-1",
          kind: "synthesize",
          status: "succeeded",
          error_code: null,
          public_message: "合成完成",
          created_at: "2026-09-06T00:00:00Z",
          started_at: null,
          finished_at: "2026-09-06T00:00:02Z",
        },
      ],
      syntheses: [
        {
          job_id: "job-1",
          voice_profile_id: "voice-1",
          status: "succeeded",
          download_ready: true,
          created_at: "2026-09-06T00:00:00Z",
        },
      ],
      adminUsers: [
        {
          id: "admin-1",
          username: "admin",
          role: "admin",
          status: "active",
          created_at: "2026-09-01T00:00:00Z",
          dataset_count: 2,
          voice_count: 3,
          job_count: 4,
        },
      ],
    });

    const navigate = () => {};
    const dashboard = render(<DashboardPage onNavigate={navigate} />);
    expect(await screen.findByText("让每一种声音，都能被清晰表达。")).toBeVisible();
    expect(dashboard.container.querySelector(".dashboard-board-grid")).toBeInTheDocument();
    dashboard.unmount();

    const create = render(<CreateVoiceWizard onNavigate={navigate} />);
    expect(create.container.querySelector(".create-upload-stage")).toBeInTheDocument();
    create.unmount();

    const library = render(<VoiceLibraryPage onNavigate={navigate} onSynthesize={() => {}} />);
    expect(await screen.findByText("我的旁白")).toBeVisible();
    expect(library.container.querySelector(".voice-library-board")).toBeInTheDocument();
    library.unmount();

    const synthesis = render(<SynthesisStudioPage onNavigate={navigate} />);
    expect(await screen.findByText("语音创作")).toBeVisible();
    expect(synthesis.container.querySelector(".synthesis-console-grid")).toBeInTheDocument();
    synthesis.unmount();

    const tasks = render(<TaskCenterPage />);
    expect(await screen.findByText("合成完成")).toBeVisible();
    expect(tasks.container.querySelector(".task-board")).toBeInTheDocument();
    tasks.unmount();

    const safety = render(<SafetyCenterPage />);
    expect(await screen.findByRole("list", { name: "安全验证流程" })).toBeVisible();
    expect(safety.container.querySelector(".safety-board")).toBeInTheDocument();
    safety.unmount();

    const admin = render(<AdminCenterPage user={{ ...USER, id: "admin-1", role: "admin" }} />);
    expect(await screen.findByRole("table", { name: "用户列表" })).toBeVisible();
    expect(admin.container.querySelector(".admin-board")).toBeInTheDocument();
  });
});
