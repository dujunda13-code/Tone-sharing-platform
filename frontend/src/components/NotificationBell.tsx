import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { NotificationItem } from "../api/types";

function BellGlyph() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.2 17.2h11.6l-1.3-1.8v-4.1a4.5 4.5 0 0 0-9 0v4.1l-1.3 1.8Z" /><path d="M10 20h4" /></svg>; }
function text(item: NotificationItem) { return item.type === "like" ? `${item.actor_display_name} 赞了你的音色《${item.voice_name}》` : `${item.actor_display_name} 评论了你的音色《${item.voice_name}》${item.comment_excerpt ? `：${item.comment_excerpt}` : ""}`; }

export function NotificationBell({ onViewAll }: { onViewAll: () => void }) {
  const [open, setOpen] = useState(false);
  const [unread, setUnread] = useState(0);
  const [items, setItems] = useState<NotificationItem[] | null>(null);
  const anchor = useRef<HTMLSpanElement | null>(null);
  useEffect(() => { let active = true; const refresh = () => void api.notificationsUnreadCount().then((result) => { if (active) setUnread(result.count); }).catch(() => {}); refresh(); const timer = window.setInterval(refresh, 30000); return () => { active = false; window.clearInterval(timer); }; }, []);
  useEffect(() => { if (!open) return; let active = true; void api.notifications({ limit: 20 }).then((result) => { if (active) setItems(result.items); }).catch(() => { if (active) setItems([]); }); return () => { active = false; }; }, [open]);
  useEffect(() => { if (!open) return; const close = (event: PointerEvent) => { if (anchor.current && !anchor.current.contains(event.target as Node)) setOpen(false); }; const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); }; document.addEventListener("pointerdown", close); document.addEventListener("keydown", escape); return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", escape); }; }, [open]);
  const markAllRead = async () => { try { await api.notificationsMarkRead({ all: true }); const result = await api.notifications({ limit: 20 }); setItems(result.items); setUnread(0); } catch {} };
  return <span className="bell-anchor" ref={anchor}><button className="nav-icon-button notification-button" type="button" aria-expanded={open} aria-label={unread > 0 ? `通知（${unread} 条未读）` : "通知"} onClick={() => setOpen((value) => !value)}><BellGlyph />{unread > 0 && <span className="notification-dot" aria-hidden="true" />}</button>{open && <div className="notification-panel card" role="dialog" aria-label="通知面板">{items === null ? <p className="hint">正在加载通知…</p> : items.length === 0 ? <p className="hint">暂无通知。</p> : items.map((item) => <div key={item.id} className={`notification-item${item.is_read ? "" : " unread"}`}>{text(item)}</div>)}<div className="notification-actions"><button type="button" className="secondary" disabled={unread === 0} onClick={() => void markAllRead()}>全部已读</button><button type="button" className="secondary" onClick={() => { setOpen(false); onViewAll(); }}>查看全部</button></div></div>}</span>;
}
