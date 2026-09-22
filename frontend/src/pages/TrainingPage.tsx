import type { WorkspacePage } from "../components/AppShell";
import { CreateVoicePage } from "./CreateVoicePage";

/** 兼容导出：旧训练页委托创建音色向导，避免双份训练逻辑。 */
export function TrainingPage() {
  return <CreateVoicePage onNavigate={(_page: WorkspacePage) => undefined} />;
}
