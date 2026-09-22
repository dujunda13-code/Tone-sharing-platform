/**
 * 工作台背景视频：丝带的波峰从左向右传递，容器位置保持固定。
 * 静音循环播放，避免干扰页面的阅读与操作。
 */
export function RibbonBackground({ className = "" }: { className?: string }) {
  return (
    <video
      className={`ribbon-background ${className}`}
      aria-hidden="true"
      autoPlay
      loop
      muted
      playsInline
      preload="auto"
      poster="/dashboard-ribbon-motion-poster.png"
      tabIndex={-1}
    >
      <source src="/dashboard-ribbon-motion.mp4" type="video/mp4" />
    </video>
  );
}
