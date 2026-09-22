import { useEffect, useState } from "react";

import { api } from "./api/client";
import type { AuthUser, HealthResponse } from "./api/types";
import { AppShell, type WorkspacePage } from "./components/AppShell";
import { AuthGate } from "./components/AuthGate";
import { resetWorkspaceDrafts } from "./state/drafts";
import type { SynthesisRequest } from "./state/drafts";
import { CreateVoicePage } from "./pages/CreateVoicePage";
import { DashboardPage } from "./pages/DashboardPage";
import { PlazaPage } from "./pages/PlazaPage";
import { ProfilePage, type ProfileTab } from "./pages/ProfilePage";
import { SafetyCenterPage } from "./pages/SafetyCenterPage";
import { SynthesisStudioPage } from "./pages/SynthesisStudioPage";
import { TaskCenterPage } from "./pages/TaskCenterPage";
import { AdminCenterPage } from "./pages/AdminCenterPage";
import { VoiceLibraryPage } from "./pages/VoiceLibraryPage";

function WorkspaceApp({ user, onLogout }: { user: AuthUser; onLogout: () => Promise<void> }) {
  const [page, setPage] = useState<WorkspacePage>("dashboard");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [synthesisRequest, setSynthesisRequest] = useState<SynthesisRequest | null>(null);
  const [profileTab, setProfileTab] = useState<ProfileTab>("posts");

  useEffect(() => {
    let active = true;
    const refresh = () =>
      void api
        .health()
        .then((next) => {
          if (active) setHealth(next);
        })
        .catch(() => {
          if (active) setHealth(null);
        });
    refresh();
    const timer = window.setInterval(refresh, 30000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const navigate = (next: WorkspacePage) => setPage(next);
  const synthesizeWith = (voiceId: string, voiceLabel?: string) => {
    setSynthesisRequest((previous) => ({ voiceId, voiceLabel, token: (previous?.token ?? 0) + 1 }));
    setPage("synthesize");
  };
  const logout = async () => {
    resetWorkspaceDrafts();
    await onLogout();
  };

  return (
    <AppShell user={user} health={health} activePage={page} onNavigate={navigate} onLogout={logout} onOpenNotifications={() => { setProfileTab("notifications"); navigate("profile"); }}>
      {page === "dashboard" && <DashboardPage onNavigate={navigate} />}
      {page === "plaza" && <PlazaPage onNavigate={navigate} onSynthesize={synthesizeWith} />}
      {page === "profile" && <ProfilePage tab={profileTab} onTabChange={setProfileTab} onNavigate={navigate} />}
      {page === "voices" && <VoiceLibraryPage onNavigate={navigate} onSynthesize={synthesizeWith} />}
      {page === "create" && <CreateVoicePage onNavigate={navigate} />}
      {page === "synthesize" && (
        <SynthesisStudioPage synthesisRequest={synthesisRequest} onNavigate={navigate} />
      )}
      {page === "tasks" && <TaskCenterPage />}
      {page === "safety" && <SafetyCenterPage />}
      {page === "admin" && user.role === "admin" && <AdminCenterPage user={user} />}
    </AppShell>
  );
}

export function App() {
  return (
    <AuthGate>
      {(user, onLogout) => <WorkspaceApp user={user} onLogout={onLogout} />}
    </AuthGate>
  );
}
