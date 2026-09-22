import { useEffect, useRef, useState } from "react";

import { api, getApiError } from "../api/client";
import type { PlazaComment, PlazaPost, VoiceSummary } from "../api/types";
import type { WorkspacePage } from "../components/AppShell";
import { AtmosphericWave } from "../components/AtmosphericWave";
import { Dropdown } from "../components/Dropdown";

type PlazaFilter = "all" | "mine" | "favorited";

const FILTER_LABELS: Array<{ id: PlazaFilter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "mine", label: "我的发布" },
  { id: "favorited", label: "我的收藏" },
];

function WaveMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 48 48">
      <path d="M8 24h4m4-9v18m8-25v32m8-25v18m8-9h4" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="2.5" />
    </svg>
  );
}

function ActionIcon({ kind }: { kind: "heart" | "bookmark" | "comment" }) {
  const path = kind === "heart" ? <path d="M20.8 8.5c0 4.2-8.8 10-8.8 10s-8.8-5.8-8.8-10a4.7 4.7 0 0 1 8.8-2.2 4.7 4.7 0 0 1 8.8 2.2Z" /> : kind === "bookmark" ? <path d="M6 3.5h12v17l-6-4-6 4v-17Z" /> : <path d="M4 5h16v12H9l-5 3V5Z" />;
  return <svg viewBox="0 0 24 24" aria-hidden="true">{path}</svg>;
}

/** 浏览、互动和本地导入已发布音色。所有实际权限校验仍在后端执行。 */
export function PlazaPage({
  onNavigate,
  onSynthesize,
}: {
  onNavigate: (page: WorkspacePage) => void;
  onSynthesize: (voiceId: string, voiceLabel?: string) => void;
}) {
  const [posts, setPosts] = useState<PlazaPost[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<PlazaFilter>("all");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<"newest" | "likes">("newest");
  const [reloadToken, setReloadToken] = useState(0);
  const [comments, setComments] = useState<Record<string, PlazaComment[]>>({});
  const [openComments, setOpenComments] = useState<Record<string, boolean>>({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});
  const previewUrlsRef = useRef<Record<string, string>>({});
  const [publishOpen, setPublishOpen] = useState(false);
  const [publishVoices, setPublishVoices] = useState<VoiceSummary[] | null>(null);
  const [publishVoiceId, setPublishVoiceId] = useState("");
  const [publishDescription, setPublishDescription] = useState("");
  const [importTarget, setImportTarget] = useState<PlazaPost | null>(null);
  const [importConsent, setImportConsent] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [pendingActions, setPendingActions] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let active = true;
    setError(null);
    void api
      .plazaPosts({
        mine: filter === "mine",
        favorited: filter === "favorited",
        q: query.trim() || undefined,
        sort,
      })
      .then((list) => {
        if (active) setPosts(list.items);
      })
      .catch((cause: unknown) => {
        if (active) {
          setPosts([]);
          setError(getApiError(cause).message);
        }
      });
    return () => {
      active = false;
    };
  }, [filter, query, sort, reloadToken]);

  useEffect(() => () => {
    Object.values(previewUrlsRef.current).forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const reload = () => {
    setPosts(null);
    setReloadToken((token) => token + 1);
  };

  const actionKey = (postId: string, action: "like" | "favorite") => `${postId}:${action}`;

  const updatePost = (postId: string, update: (post: PlazaPost) => PlazaPost) => {
    setPosts((previous) => previous?.map((item) => (item.id === postId ? update(item) : item)) ?? previous);
  };

  const setActionPending = (postId: string, action: "like" | "favorite", pending: boolean) => {
    const key = actionKey(postId, action);
    setPendingActions((previous) => {
      if (pending) return { ...previous, [key]: true };
      const { [key]: _, ...rest } = previous;
      return rest;
    });
  };

  const toggleLike = async (post: PlazaPost) => {
    const key = actionKey(post.id, "like");
    if (pendingActions[key]) return;
    const liked = !post.liked_by_me;
    const delta = liked ? 1 : -1;
    setNotice(null);
    setActionPending(post.id, "like", true);
    updatePost(post.id, (item) => ({ ...item, liked_by_me: liked, like_count: Math.max(0, item.like_count + delta) }));
    try {
      if (post.liked_by_me) await api.plazaUnlike(post.id);
      else await api.plazaLike(post.id);
    } catch (cause) {
      updatePost(post.id, (item) => ({ ...item, liked_by_me: post.liked_by_me, like_count: Math.max(0, item.like_count - delta) }));
      setNotice(getApiError(cause).message);
    } finally {
      setActionPending(post.id, "like", false);
    }
  };

  const toggleFavorite = async (post: PlazaPost) => {
    const key = actionKey(post.id, "favorite");
    if (pendingActions[key]) return;
    const favorited = !post.favorited_by_me;
    const delta = favorited ? 1 : -1;
    setNotice(null);
    setActionPending(post.id, "favorite", true);
    updatePost(post.id, (item) => ({ ...item, favorited_by_me: favorited, favorite_count: Math.max(0, item.favorite_count + delta) }));
    try {
      if (post.favorited_by_me) await api.plazaUnfavorite(post.id);
      else await api.plazaFavorite(post.id);
      if (!favorited && filter === "favorited") {
        setPosts((previous) => previous?.filter((item) => item.id !== post.id) ?? previous);
      }
    } catch (cause) {
      updatePost(post.id, (item) => ({ ...item, favorited_by_me: post.favorited_by_me, favorite_count: Math.max(0, item.favorite_count - delta) }));
      setNotice(getApiError(cause).message);
    } finally {
      setActionPending(post.id, "favorite", false);
    }
  };

  const loadComments = async (postId: string) => {
    const response = await api.plazaComments(postId);
    setComments((previous) => ({ ...previous, [postId]: response.items }));
  };

  const toggleComments = async (postId: string) => {
    if (openComments[postId]) {
      setOpenComments((previous) => ({ ...previous, [postId]: false }));
      return;
    }
    try {
      await loadComments(postId);
      setOpenComments((previous) => ({ ...previous, [postId]: true }));
    } catch (cause) {
      setNotice(getApiError(cause).message);
    }
  };

  const addComment = async (postId: string) => {
    const content = (drafts[postId] ?? "").trim();
    if (!content || content.length > 500) return;
    try {
      await api.plazaAddComment(postId, content);
      setDrafts((previous) => ({ ...previous, [postId]: "" }));
      await loadComments(postId);
      reload();
    } catch (cause) {
      setNotice(getApiError(cause).message);
    }
  };

  const loadPreview = async (postId: string) => {
    if (previewUrls[postId]) return;
    try {
      const response = await fetch(api.plazaPreviewUrl(postId), { credentials: "include" });
      if (!response.ok) throw new Error("preview failed");
      const url = URL.createObjectURL(await response.blob());
      setPreviewUrls((previous) => {
        const next = { ...previous, [postId]: url };
        previewUrlsRef.current = next;
        return next;
      });
    } catch {
      setNotice("试听加载失败，请稍后重试");
    }
  };

  const openPublish = async () => {
    setPublishOpen(true);
    setPublishVoices(null);
    setPublishVoiceId("");
    setPublishDescription("");
    try {
      const [voiceResponse, postResponse] = await Promise.all([api.voices(), api.plazaPosts({ mine: true })]);
      const publishedIds = new Set(postResponse.items.map((post) => post.voice.voice_profile_id));
      const eligible = voiceResponse.items.filter(
        (voice) => voice.status === "ready" && voice.can_synthesize && !publishedIds.has(voice.id),
      );
      setPublishVoices(eligible);
      setPublishVoiceId(eligible[0]?.id ?? "");
    } catch (cause) {
      setPublishOpen(false);
      setNotice(getApiError(cause).message);
    }
  };

  const publish = async () => {
    if (!publishVoiceId || publishDescription.length > 500) return;
    try {
      await api.plazaPublish({ voice_profile_id: publishVoiceId, description: publishDescription.trim() || undefined });
      setPublishOpen(false);
      setNotice("音色已发布到广场");
      reload();
    } catch (cause) {
      setNotice(getApiError(cause).message);
    }
  };

  const openImport = (post: PlazaPost) => {
    setImportTarget(post);
    setImportConsent(false);
  };

  const importVoice = async () => {
    if (!importTarget || !importConsent) return;
    try {
      await api.plazaImport(importTarget.id, true);
      setImportTarget(null);
      setNotice("已导入到我的音色");
    } catch (cause) {
      setNotice(getApiError(cause).message);
    }
  };

  return (
    <section className="preview-page page-stack">
      <header className="plaza-hero atmospheric-panel tone-mint">
        <AtmosphericWave className="plaza-hero-waves" />
        <div className="plaza-hero-copy">
          <p className="plaza-hero-eyebrow">音色广场</p>
          <h1>让声音，被更多人听见</h1>
          <p className="card-lede">浏览所有已发布的音色档案；试听、收藏，或将喜欢的声音用于创作。</p>
          <div className="plaza-hero-actions">
            <button type="button" className="primary" onClick={() => void openPublish()}>发布我的音色</button>
            <button type="button" className="secondary" onClick={() => onNavigate("create")}>去创建音色</button>
          </div>
        </div>
      </header>
      {notice && <div className="card" role="status">{notice}</div>}
      <div className="card plaza-toolbar plaza-toolbar-card">
        <div className="plaza-filter-tabs" role="tablist" aria-label="广场过滤">
          {FILTER_LABELS.map((item) => (
            <button key={item.id} type="button" role="tab" aria-selected={filter === item.id} className={filter === item.id ? "secondary active-filter" : "secondary"} onClick={() => setFilter(item.id)}>
              {item.label}
            </button>
          ))}
        </div>
        <input className="plaza-search" type="search" placeholder="搜索音色名称或简介" value={query} onChange={(event) => setQuery(event.target.value)} aria-label="搜索音色" />
        <Dropdown className="plaza-sort" label="排序方式" value={sort} onChange={(value) => setSort(value as "newest" | "likes")} options={[{ value: "newest", label: "最新发布" }, { value: "likes", label: "最多点赞" }]} />
        <button type="button" className="primary" onClick={() => void openPublish()}>发布音色</button>
      </div>
      {error && <div className="card" role="alert"><p>{error}</p><button type="button" className="secondary" onClick={reload}>重试</button></div>}
      {!error && posts === null && <section className="card" aria-live="polite">正在加载音色广场…</section>}
      {!error && posts !== null && posts.length === 0 && (
        <section className="card plaza-empty-state" aria-label="广场空状态">
          <div className="plaza-empty-wave"><WaveMark /></div>
          <h2>还没有公开音色</h2>
          <p>你的声音创作正在等待被分享。发布第一条音色，让更多人听见它。</p>
          <div className="plaza-empty-actions">
            <button type="button" className="primary" onClick={() => void openPublish()}>发布我的音色</button>
            <button type="button" className="secondary" onClick={() => onNavigate("create")}>去创建音色</button>
          </div>
          <div className="plaza-empty-benefits" aria-label="广场功能说明">
            <span>分享你的声音</span>
            <span>遇见同好</span>
            <span>激发更多创作</span>
          </div>
        </section>
      )}
      {posts !== null && posts.length > 0 && (
        <section className="plaza-directory" role="list" aria-label="已发布音色">
          {posts.map((post) => {
            const draft = drafts[post.id] ?? "";
            const likePending = Boolean(pendingActions[actionKey(post.id, "like")]);
            const favoritePending = Boolean(pendingActions[actionKey(post.id, "favorite")]);
            return (
              <article key={post.id} role="listitem" className="plaza-directory-row">
                <div className="plaza-directory-main">
                  <div className="plaza-post-identity">
                    <span className="plaza-voice-mark" aria-hidden="true"><WaveMark /></span>
                    <div>
                      <h2 className="plaza-voice-name">{post.voice.display_name}</h2>
                      <span className="plaza-card-author">创作者 · <span>{post.author.display_name}</span> · {post.voice.reference_count} 段参考</span>
                    </div>
                  </div>
                  {post.voice.reference_emotions.length > 0 && <div className="plaza-emotion-tags">{post.voice.reference_emotions.map((emotion) => <span key={emotion} className="status-badge">{emotion}</span>)}</div>}
                  {post.description && <p className="plaza-description">{post.description}</p>}
                </div>
                <div className="plaza-directory-preview">
                  {previewUrls[post.id] ? <audio aria-label={`试听 ${post.voice.display_name}`} controls preload="none" src={previewUrls[post.id]} /> : <button type="button" className="secondary" onClick={() => void loadPreview(post.id)}>试听</button>}
                </div>
                <div className="plaza-directory-interactions">
                  <div className="plaza-actions">
                    <button type="button" aria-label={post.liked_by_me ? "取消点赞" : "点赞"} aria-pressed={post.liked_by_me} className={post.liked_by_me ? "plaza-icon-action selected" : "plaza-icon-action"} disabled={likePending} onClick={() => void toggleLike(post)}><ActionIcon kind="heart" /><span>{post.like_count}</span></button>
                    <button type="button" aria-label={post.favorited_by_me ? "取消收藏" : "收藏"} aria-pressed={post.favorited_by_me} className={post.favorited_by_me ? "plaza-icon-action selected" : "plaza-icon-action"} disabled={favoritePending} onClick={() => void toggleFavorite(post)}><ActionIcon kind="bookmark" /><span>{post.favorite_count}</span></button>
                    <button type="button" className="plaza-icon-action" aria-expanded={Boolean(openComments[post.id])} onClick={() => void toggleComments(post.id)}><ActionIcon kind="comment" /><span>评论 {post.comment_count}</span></button>
                    <button type="button" className="primary" onClick={() => onSynthesize(post.voice.voice_profile_id, post.voice.display_name)}>立即创作</button>
                  </div>
                  <div className="plaza-directory-tools">
                    <a href={api.plazaDownloadUrl(post.id)} download>下载音色包</a>
                    <a href={api.plazaReferenceDownloadUrl(post.id)} download>主参考音频</a>
                    <button type="button" onClick={() => openImport(post)}>导入到我的音色</button>
                  </div>
                </div>
                {openComments[post.id] && (
                  <section className="plaza-comments" aria-label="评论区">
                    <header className="plaza-comments-heading"><h3>评论</h3><span>{post.comment_count} 条讨论</span></header>
                    {(comments[post.id] ?? []).length === 0 ? <p className="plaza-comments-empty">还没有评论，来分享你的想法。</p> : <div className="plaza-comment-list">{(comments[post.id] ?? []).map((comment) => <div key={comment.id} className="plaza-comment"><span className="plaza-comment-avatar" aria-hidden="true">{comment.author.display_name.slice(0, 1)}</span><div><div className="plaza-comment-meta">{comment.author.display_name}</div><p>{comment.content}</p></div></div>)}</div>}
                    <div className="plaza-comment-compose"><textarea aria-label="发表评论" placeholder="写下你的评论…" value={draft} maxLength={500} onChange={(event) => setDrafts((previous) => ({ ...previous, [post.id]: event.target.value }))} /><div><span>{draft.length}/500</span><button type="button" className="primary" disabled={!draft.trim() || draft.length > 500} onClick={() => void addComment(post.id)}>发表评论</button></div></div>
                  </section>
                )}
              </article>
            );
          })}
        </section>
      )}
      {publishOpen && (
        <div className="plaza-modal-backdrop" role="presentation">
          <section className="card plaza-modal-content" role="dialog" aria-modal="true" aria-label="发布音色">
            <div className="plaza-modal-heading"><h2>发布音色</h2><button type="button" className="secondary" onClick={() => setPublishOpen(false)}>关闭</button></div>
            {publishVoices === null ? <p>正在加载可发布音色…</p> : publishVoices.length === 0 ? <p>没有可发布的音色。</p> : <><p className="card-lede">选择一个已就绪的音色，写一段简介，让更多人了解它。</p><div className="field"><label>选择音色</label><Dropdown label="发布音色选择" value={publishVoiceId} onChange={setPublishVoiceId} options={publishVoices.map((voice) => ({ value: voice.id, label: voice.display_name }))} /></div><div className="field"><label htmlFor="publish-description">音色简介</label><textarea id="publish-description" value={publishDescription} maxLength={500} placeholder="介绍音色特点与适用场景…" onChange={(event) => setPublishDescription(event.target.value)} /><span className="plaza-publish-count">{publishDescription.length}/500</span></div><div className="plaza-modal-footer"><button type="button" className="secondary" onClick={() => setPublishOpen(false)}>取消</button><button type="button" className="primary" onClick={() => void publish()}>发布音色</button></div></>}
          </section>
        </div>
      )}
      {importTarget && (
        <div className="plaza-modal-backdrop" role="presentation">
          <section className="card plaza-modal-content" role="dialog" aria-modal="true" aria-label="导入音色">
            <div className="plaza-modal-heading"><h2>导入「{importTarget.voice.display_name}」</h2><button type="button" className="secondary" onClick={() => setImportTarget(null)}>关闭</button></div>
            <label className="consent-box"><input type="checkbox" checked={importConsent} onChange={(event) => setImportConsent(event.target.checked)} />我确认已获得该音色的使用授权</label>
            <button type="button" className="primary" disabled={!importConsent} onClick={() => void importVoice()}>确认导入</button>
          </section>
        </div>
      )}
    </section>
  );
}
