import { CreateVoiceWizard } from "../components/CreateVoiceWizard";
import type { WorkspacePage } from "../components/AppShell";

export function CreateVoicePage({ onNavigate }: { onNavigate: (page: WorkspacePage) => void }) {
  return (
    <div className="page-stack preview-page create-voice-page">
      <section className="card page-intro">
        <h2>创建音色</h2>
        <p className="card-lede">
          为你的声音命名，上传一段已获授权的参考音频；系统完成分析与审核后，音色立即可用于语音创作。
        </p>
      </section>
      <CreateVoiceWizard onNavigate={onNavigate} />
    </div>
  );
}
