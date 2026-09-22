import { useCallback, useEffect, useState } from "react";

import { api, getApiError } from "../api/client";
import type { SynthesisSummary, WatermarkResponse } from "../api/types";
import type { SafeError } from "../components/ErrorPanel";
import { ErrorPanel } from "../components/ErrorPanel";
import { EmptyState } from "../components/EmptyState";
import { SafetySummary } from "../components/SafetySummary";

/** 用户可理解的安全验证阶段：与后端 fail-closed 检查链一一对应。 */
const SAFETY_STAGES = ["授权确认", "内容检查", "来源水印", "声音指纹", "结果判定"];

/** 安全中心：已发布合成结果的水印复检、指纹与下载入口；结论完全来自真实检测结果。 */
export function SafetyCenterPage() {
  const [items, setItems] = useState<SynthesisSummary[] | null>(null);
  const [error, setError] = useState<SafeError | null>(null);
  const [verifications, setVerifications] = useState<Record<string, WatermarkResponse>>({});
  const [verifyingJob, setVerifyingJob] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    let active = true;
    void api
      .synthesisList()
      .then((list) => {
        if (!active) return;
        setItems(list.items);
        setError(null);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setItems(null);
        setError(getApiError(requestError));
      });
    return () => {
      active = false;
    };
  }, [reloadToken]);

  const verify = async (jobId: string) => {
    setVerifyingJob(jobId);
    try {
      const result = await api.detectWatermark(jobId);
      setVerifications((previous) => ({ ...previous, [jobId]: result }));
    } catch (requestError) {
      setError(getApiError(requestError));
    } finally {
      setVerifyingJob(null);
    }
  };

  if (items === null && error === null) {
    return (
      <section className="card" aria-live="polite">
        <h2>正在加载合成记录…</h2>
      </section>
    );
  }

  if (error && items === null) {
    return (
      <section className="card">
        <h2>安全中心</h2>
        <ErrorPanel error={error} hint="合成列表接口由后端提供；未实现前保持失败提示。" />
        <button className="secondary" type="button" onClick={retry}>
          重试
        </button>
      </section>
    );
  }

  if (items !== null && items.length === 0) {
    return (
      <section className="card">
        <h2>安全中心</h2>
        <EmptyState title="还没有合成记录" description="完成一次语音合成后，可以在这里复检音频来源。" />
      </section>
    );
  }

  return (
    <div className="page-stack preview-page safety-center-page">
      <section className="card page-intro">
        <h2>安全中心</h2>
        <p className="card-lede">
          平台生成的音频均嵌入来源水印并记录声音指纹；可随时复检每条结果的来源。严重外部处理后可能无法确认来源，属正常安全行为。
        </p>
        <ol className="safety-stages" aria-label="安全验证流程">
          {SAFETY_STAGES.map((stage) => (
            <li key={stage}>
              <span className="stage-title">{stage}</span>
            </li>
          ))}
        </ol>
      </section>
      <div className="safety-board">
        <div>
          {error && <ErrorPanel error={error} />}
          <ul className="safety-list">
            {items?.map((item) => (
              <li key={item.job_id}>
                <SafetySummary
                  item={item}
                  verification={verifications[item.job_id]}
                  verifying={verifyingJob === item.job_id}
                  onVerify={(jobId) => void verify(jobId)}
                />
              </li>
            ))}
          </ul>
        </div>
        <aside className="card safety-protection-card" aria-label="账号安全状态">
          <span className="safety-shield" aria-hidden="true">⌁</span>
          <h3>你的声音已受到保护</h3>
          <p>每次合成都会经过来源校验与内容检查，结果仅在通过全部检查后发布。</p>
          <span className="safety-card-glow" aria-hidden="true" />
        </aside>
      </div>
    </div>
  );
}
