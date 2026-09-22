function similarityPercent(similarity: number | null | undefined): string | null {
  return similarity == null ? null : `${(similarity * 100).toFixed(1)}%`;
}

/**
 * 合成结果卡：波形装饰、播放器、质量提示与下载。
 * 仅在 download_ready 时渲染；低相似度提示质量但不阻止下载。
 */
export function AudioResultCard({
  src,
  downloadName,
  similarity,
  warningCodes = [],
}: {
  src: string;
  downloadName: string;
  similarity?: number | null;
  warningCodes?: string[];
}) {
  const percent = similarityPercent(similarity);
  const lowSimilarity = warningCodes.includes("SPEAKER_SIMILARITY_BELOW_RECOMMENDED");
  return (
    <div className="audio-result-card">
      <div className="waveform" aria-hidden="true">
        {Array.from({ length: 28 }, (_, index) => (
          <span key={index} />
        ))}
      </div>
      <audio controls preload="none" src={src} />
      {percent && (
        <p className="note">
          音色相似度：{percent}
          {lowSimilarity ? "（建议值 90%）" : ""}
        </p>
      )}
      {lowSimilarity && (
        <p className="quality-warning compact" role="alert">
          相似度 {percent}，低于建议值 90%：该结果可用但音色还原可能不佳，可重新录制更清晰的参考后再次合成。
        </p>
      )}
      <a
        className="download-link"
        role="button"
        href={src}
        download={downloadName}
      >
        下载音频
      </a>
    </div>
  );
}
