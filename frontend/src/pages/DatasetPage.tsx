import type { WorkspacePage } from "../components/AppShell";
import { CreateVoicePage } from "./CreateVoicePage";

/** 兼容导出：旧数据质检页委托创建音色向导，避免双份上传/质检逻辑。 */
export function DatasetPage() {
  return <CreateVoicePage onNavigate={(_page: WorkspacePage) => undefined} />;
}
