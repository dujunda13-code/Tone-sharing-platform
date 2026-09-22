const LOW_SNR = "SNR_BELOW_RECOMMENDED";

export function QualityWarning({
  warningCodes = [],
  snrDb,
  thresholdDb = 20,
  compact = false,
}: {
  warningCodes?: readonly string[] | null;
  snrDb: number | null | undefined;
  thresholdDb?: number;
  compact?: boolean;
}) {
  if (!warningCodes?.includes(LOW_SNR)) return null;
  const measured = snrDb === null || snrDb === undefined ? "未知" : `${snrDb.toFixed(2)} dB`;
  return (
    <div className={`quality-warning${compact ? " compact" : ""}`} role="status">
      <strong>参考音频信噪比较低</strong>
      <span>实测 {measured}，低于建议值 {thresholdDb.toFixed(0)} dB；仍可继续，但可能降低音色相似度和清晰度。</span>
    </div>
  );
}
