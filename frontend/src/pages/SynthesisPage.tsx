import type { WorkspacePage } from "../components/AppShell";
import { SynthesisStudioPage } from "./SynthesisStudioPage";

/** 兼容导出：旧合成页委托新的语音创作工作台。 */
export function SynthesisPage() {
  return <SynthesisStudioPage onNavigate={(_page: WorkspacePage) => undefined} />;
}
