const BAR_HEIGHTS = [
  9, 17, 11, 6, 8, 13, 19, 16, 29, 42, 57, 79, 94, 72, 60, 48, 36,
  27, 22, 31, 42, 54, 71, 87, 100, 79, 62, 48, 35, 29, 40, 52, 68,
  56, 41, 29, 21, 15, 13, 20, 14, 10, 18, 31, 53, 77, 59, 41, 28,
  19, 27, 15,
];

/** 装饰性波形，与音频播放状态无关。 */
export function AudioWaveform({ className = "" }: { className?: string }) {
  return (
    <div className={`audio-waveform ${className}`.trim()} aria-hidden="true">
      {BAR_HEIGHTS.map((height, index) => (
        <span key={index} style={{ height: `${height}%` }} />
      ))}
    </div>
  );
}
