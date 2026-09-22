import { useCallback, useEffect, useState } from "react";

import { api, getApiError } from "../api/client";
import type { VoiceSummary } from "../api/types";
import type { WorkspacePage } from "../components/AppShell";
import type { SafeError } from "../components/ErrorPanel";
import { ErrorPanel } from "../components/ErrorPanel";
import { EmptyState } from "../components/EmptyState";
import { VoiceCard } from "../components/VoiceCard";
import { AddReferenceModal } from "../components/AddReferenceModal";

/** 我的音色：仅显示当前用户的真实音色档案；只有可用音色才能进入合成。 */
export function VoiceLibraryPage({
  onNavigate,
  onSynthesize,
}: {
  onNavigate: (page: WorkspacePage) => void;
  onSynthesize: (voiceId: string) => void;
}) {
  const [voices, setVoices] = useState<VoiceSummary[] | null>(null);
  const [error, setError] = useState<SafeError | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [selectedVoiceForAdd, setSelectedVoiceForAdd] = useState<VoiceSummary | null>(null);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);

  const handleDeleteReference = useCallback(
    async (profileId: string, referenceId: string) => {
      try {
        await api.deleteVoiceReference(profileId, referenceId);
        setReloadToken((token) => token + 1);
      } catch (requestError: unknown) {
        setError(getApiError(requestError));
      }
    },
    []
  );

  useEffect(() => {
    let active = true;
    void api
      .voices()
      .then((list) => {
        if (!active) return;
        setVoices(list.items);
        setError(null);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setVoices(null);
        setError(getApiError(requestError));
      });
    return () => {
      active = false;
    };
  }, [reloadToken]);

  if (voices === null && error === null) {
    return (
      <section className="card" aria-live="polite">
        <h2>正在加载音色档案…</h2>
        <p className="card-lede">音色列表来自你的账号数据，很快就会显示。</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="card">
        <h2>我的音色</h2>
        <ErrorPanel error={error} hint="音色列表接口由后端提供；未实现前保持失败提示。" />
        <button className="secondary" type="button" onClick={retry}>
          重试
        </button>
      </section>
    );
  }

  if (voices && voices.length === 0) {
    return (
      <section className="card">
        <h2>我的音色</h2>
        <EmptyState
          title="还没有音色档案"
          description="上传授权的 3–10 秒参考音频并完成审核后，零样本音色会出现在这里。"
          action={
            <button className="primary" type="button" onClick={() => onNavigate("create")}>
              去创建音色
            </button>
          }
        />
      </section>
    );
  }

  return (
    <div className="page-stack preview-page voice-library-page">
      <section className="card page-intro voice-library-intro">
        <div>
          <h2>我的音色</h2>
          <p className="card-lede">仅显示属于当前账号的音色档案；只有真实可用的音色才能进入合成流程。</p>
        </div>
        <button className="primary" type="button" onClick={() => onNavigate("create")}>
          ＋ 创建音色
        </button>
      </section>
      <section className="card voice-library-board" aria-label="音色列表">
        <div className="voice-library-toolbar">
          <div className="voice-filter-pills" aria-label="音色筛选">
            <button className="secondary active-filter" type="button">全部</button>
            <button className="secondary" type="button">女声</button>
            <button className="secondary" type="button">男声</button>
            <button className="secondary" type="button">叙事</button>
            <button className="secondary" type="button">商务</button>
          </div>
          <label className="voice-search-field">
            <span className="sr-only-label">搜索音色</span>
            <input type="search" placeholder="搜索音色名称、标签或描述…" />
          </label>
        </div>
        <div className="voice-library-columns" aria-hidden="true">
          <span>音色名称</span><span>标签</span><span>状态</span><span>创建时间</span><span>操作</span>
        </div>
        <div className="page-grid voice-library-rows">
        {voices?.map((voice) => (
          <VoiceCard
            key={voice.id}
            voice={voice}
            onSynthesize={onSynthesize}
            onNavigate={onNavigate}
            onAddReference={(v) => setSelectedVoiceForAdd(v)}
            onDeleteReference={handleDeleteReference}
          />
        ))}
        </div>
      </section>

      {selectedVoiceForAdd && (
        <AddReferenceModal
          voice={selectedVoiceForAdd}
          onClose={() => setSelectedVoiceForAdd(null)}
          onSuccess={() => {
            setSelectedVoiceForAdd(null);
            setReloadToken((t) => t + 1);
          }}
        />
      )}
    </div>
  );
}
