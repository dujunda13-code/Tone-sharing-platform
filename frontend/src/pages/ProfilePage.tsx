import { useEffect, useRef, useState } from "react";

import { api, getApiError } from "../api/client";
import type { NotificationItem, PlazaComment, PlazaPost, UserProfileSummary } from "../api/types";
import type { WorkspacePage } from "../components/AppShell";

export type ProfileTab = "posts" | "favorites" | "comments" | "notifications";

const TABS: Array<{ id: ProfileTab; label: string }> = [
  { id: "posts", label: "我的发布" },
  { id: "favorites", label: "我的收藏" },
  { id: "comments", label: "我的评论" },
  { id: "notifications", label: "通知历史" },
];

function notificationText(item: NotificationItem) {
  return item.type === "like"
    ? `${item.actor_display_name} 赞了你的音色《${item.voice_name}》`
    : `${item.actor_display_name} 评论了你的音色《${item.voice_name}》${item.comment_excerpt ? `：${item.comment_excerpt}` : ""}`;
}

export function ProfilePage({ tab, onTabChange, onNavigate }: { tab: ProfileTab; onTabChange: (tab: ProfileTab) => void; onNavigate: (page: WorkspacePage) => void }) {
  const [profile, setProfile] = useState<UserProfileSummary | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [bio, setBio] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [posts, setPosts] = useState<PlazaPost[]>([]);
  const [comments, setComments] = useState<PlazaComment[]>([]);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const reloadProfile = () => void api.userProfile().then((next) => {
    setProfile(next);
    setDisplayName(next.display_name);
    setBio(next.bio ?? "");
  }).catch((cause: unknown) => setNotice(getApiError(cause).message));

  useEffect(reloadProfile, []);
  useEffect(() => {
    if (tab === "posts" || tab === "favorites") {
      void api.plazaPosts(tab === "posts" ? { mine: true } : { favorited: true }).then((result) => setPosts(result.items)).catch(() => setPosts([]));
    } else if (tab === "comments") {
      void api.myPlazaComments().then((result) => setComments(result.items)).catch(() => setComments([]));
    } else {
      void api.notifications({ limit: 50 }).then((result) => setNotifications(result.items)).catch(() => setNotifications([]));
    }
  }, [tab]);

  const saveProfile = async () => {
    try {
      await api.updateUserProfile({ display_name: displayName.trim(), bio: bio.trim() || null });
      reloadProfile();
      setNotice("资料已保存");
    } catch (cause) {
      setNotice(getApiError(cause).message);
    }
  };

  const uploadAvatar = async (file: File) => {
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
      setNotice("头像仅支持 PNG、JPEG 或 WebP");
      return;
    }
    if (file.size > 2 * 1024 * 1024) {
      setNotice("头像文件需在 2MB 以内");
      return;
    }
    try {
      await api.uploadAvatar(file);
      reloadProfile();
      setNotice("头像已更新");
    } catch (cause) {
      setNotice(getApiError(cause).message);
    }
  };

  const markAllRead = async () => {
    try {
      await api.notificationsMarkRead({ all: true });
      const result = await api.notifications({ limit: 50 });
      setNotifications(result.items);
    } catch (cause) {
      setNotice(getApiError(cause).message);
    }
  };

  const canSave = displayName.trim().length > 0 && displayName.length <= 32 && bio.length <= 200;
  const isVoiceActivity = tab === "posts" || tab === "favorites";

  return <section className="preview-page page-stack">
    <header className="page-intro"><h1>个人中心</h1><p className="card-lede">管理个人资料，并查看你在音色广场的活动。</p></header>
    {notice && <div className="card" role="status">{notice}</div>}
    {profile && <section className="card profile-hero">
      <div className="profile-identity-panel">
        {profile.has_avatar ? <img className="profile-avatar" src={api.avatarUrl(profile.user_id)} alt={`${profile.display_name} 的头像`} /> : <span className="profile-avatar-fallback" aria-hidden="true">{profile.username.slice(0, 1).toUpperCase()}</span>}
        <div className="profile-identity-copy">
          <p>个人资料</p>
          <strong>{profile.username}</strong>
          <span>{profile.display_name}</span>
        </div>
        <div className="profile-stats" aria-label="个人统计"><span>发布 {profile.stats.published}</span><span>获赞 {profile.stats.likes_received}</span></div>
        <input ref={inputRef} aria-label="上传头像" type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) void uploadAvatar(file); }} />
        <button className="secondary profile-avatar-button" type="button" onClick={() => inputRef.current?.click()}>上传头像</button>
      </div>
      <div className="profile-edit-panel">
        <div className="profile-edit-heading"><h2>编辑资料</h2><p>让大家更容易认识你的声音。</p></div>
        <label className="field">昵称<input aria-label="昵称" type="text" value={displayName} maxLength={32} onChange={(event) => setDisplayName(event.target.value)} /></label>
        <label className="field">简介<textarea value={bio} maxLength={200} onChange={(event) => setBio(event.target.value)} /></label>
        <div className="profile-edit-footer"><span className="profile-character-count">{bio.length}/200</span><button className="primary" type="button" disabled={!canSave} onClick={() => void saveProfile()}>保存资料</button></div>
      </div>
    </section>}
    <section className="card profile-activity-card">
      <div className="profile-tabs" role="tablist" aria-label="个人活动">{TABS.map((item) => <button key={item.id} className={tab === item.id ? "secondary active-filter" : "secondary"} type="button" role="tab" aria-selected={tab === item.id} onClick={() => onTabChange(item.id)}>{item.label}</button>)}</div>
      {isVoiceActivity && (posts.length === 0 ? <div className="profile-empty-state"><h2>{tab === "posts" ? "还没有发布过音色" : "还没有收藏音色"}</h2><p>{tab === "posts" ? "创建第一条音色档案，分享属于你的声音。" : "去音色广场试听并收藏喜欢的声音。"}</p><button type="button" className="primary" onClick={() => onNavigate("create")}>去创建音色</button></div> : <div className="profile-activity-list">{posts.map((post) => <article key={post.id} className="profile-history-row"><span>{post.voice.display_name}</span><span>{post.author.display_name}</span><button type="button" className="secondary" onClick={() => onNavigate("plaza")}>去广场查看</button></article>)}</div>)}
      {tab === "comments" && (comments.length === 0 ? <div className="profile-empty-state"><h2>暂无评论记录</h2><p>你在音色广场留下的评论会出现在这里。</p></div> : <div className="profile-activity-list">{comments.map((comment) => <article key={comment.id} className="profile-history-row"><span>{comment.voice_name ?? "音色"}</span><span>{comment.content}</span></article>)}</div>)}
      {tab === "notifications" && <><div className="profile-section-heading"><h2>通知历史</h2><button className="secondary" type="button" onClick={() => void markAllRead()}>全部已读</button></div>{notifications.length === 0 ? <div className="profile-empty-state"><h2>暂无通知记录</h2><p>新的点赞和评论会在这里提醒你。</p></div> : <div className="profile-activity-list">{notifications.map((item) => <article key={item.id} className="profile-history-row"><span>{notificationText(item)}</span></article>)}</div>}</>}
    </section>
  </section>;
}
