import type { VoiceSummary } from "../api/types";
import { Dropdown } from "./Dropdown";

const EMOTION_LABEL_TEXT: Record<string, string> = {
  neutral: "中性",
  happy: "高兴",
  sad: "悲伤",
  angry: "生气",
  fearful: "恐惧",
  disgusted: "厌恶",
  surprised: "惊讶",
  other: "其他",
};

/** 音色选择器：只提供后端确认可合成（ready 且 can_synthesize）的零样本音色。 */
export function VoicePicker({
  voices,
  value,
  onChange,
}: {
  voices: VoiceSummary[];
  value: string;
  onChange: (voiceId: string) => void;
}) {
  const selectable = voices.filter((voice) => voice.can_synthesize && voice.status === "ready");
  return (
    <div className="field">
      <label>选择音色</label>
      <Dropdown label="选择音色" value={value} onChange={onChange} disabled={selectable.length === 0} options={[{ value: "", label: selectable.length === 0 ? "暂无可用音色" : "请选择音色", disabled: true }, ...selectable.map((voice) => ({ value: voice.id, label: voice.display_name }))]} />
      <p className="hint">只显示审核完成且系统就绪的音色；选择使用音色名称，档案 ID 保持不变。</p>
    </div>
  );
}

export { EMOTION_LABEL_TEXT };
