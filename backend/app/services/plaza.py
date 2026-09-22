from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Engine, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import (
    Notification,
    PlazaComment,
    PlazaFavorite,
    PlazaLike,
    PlazaPost,
    User,
    UserProfile,
    VoiceProfile,
    VoiceReference,
)
from backend.app.db.session import create_database_engine, init_db, session_factory
from backend.app.services.notifications import NotificationStore
from backend.app.services.sensitive_filter import SensitiveFilter
from backend.app.services.voice_profiles import public_voice_name

MAX_DESCRIPTION_LENGTH = 500
MAX_COMMENT_LENGTH = 500


class PlazaError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PlazaPostRecord:
    id: str
    author_user_id: str
    voice_profile_id: str
    description: str | None
    created_at: datetime


@dataclass(frozen=True)
class PlazaPostView:
    post: PlazaPostRecord
    voice_name: str
    voice_status: str
    reference_count: int
    reference_emotions: tuple[str, ...]
    author_display_name: str
    author_has_avatar: bool
    like_count: int
    comment_count: int
    favorite_count: int
    liked_by_viewer: bool
    favorited_by_viewer: bool


@dataclass(frozen=True)
class PlazaCommentView:
    id: str
    post_id: str
    author_user_id: str
    author_display_name: str
    author_has_avatar: bool
    content: str
    created_at: datetime
    voice_name: str | None = None


def _require_post(session: Session, post_id: str) -> PlazaPost:
    row = session.get(PlazaPost, post_id)
    if row is None:
        raise KeyError(post_id)
    return row


class PlazaStore:
    """Persist voice-plaza posts and interactions with publish gating."""

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        sensitive_filter: SensitiveFilter,
        notifications: NotificationStore | None = None,
    ) -> None:
        self.engine = engine or create_database_engine("sqlite:///data/app.db")
        init_db(self.engine)
        self._session_factory: sessionmaker[Session] = session_factory(self.engine)
        self.sensitive_filter = sensitive_filter
        self.notifications = notifications

    @staticmethod
    def _post_record(row: PlazaPost) -> PlazaPostRecord:
        return PlazaPostRecord(
            id=row.id,
            author_user_id=row.author_user_id,
            voice_profile_id=row.voice_profile_id,
            description=row.description,
            created_at=row.created_at,
        )

    def publish(
        self, author_user_id: str, *, voice_profile_id: str, description: str | None = None
    ) -> PlazaPostRecord:
        normalized = (description or "").strip() or None
        if normalized is not None and len(normalized) > MAX_DESCRIPTION_LENGTH:
            raise PlazaError("PLAZA_DESCRIPTION_TOO_LONG", f"简介最长{MAX_DESCRIPTION_LENGTH} 字")
        with self._session_factory() as session:
            profile = session.get(VoiceProfile, voice_profile_id)
            if profile is None or profile.owner_user_id != author_user_id:
                raise KeyError(voice_profile_id)
            if profile.status != "ready":
                raise PlazaError("PLAZA_VOICE_PROFILE_NOT_READY", "仅 ready 音色档案可发布")
            existing = session.scalar(
                select(PlazaPost).where(PlazaPost.voice_profile_id == voice_profile_id)
            )
            if existing is not None:
                raise PlazaError("PLAZA_POST_ALREADY_PUBLISHED", "该音色档案已发布过")
            row = PlazaPost(
                id=uuid4().hex,
                author_user_id=author_user_id,
                voice_profile_id=voice_profile_id,
                description=normalized,
            )
            session.add(row)
            session.commit()
            return self._post_record(row)

    def get_post(self, post_id: str) -> PlazaPostRecord:
        with self._session_factory() as session:
            row = session.get(PlazaPost, post_id)
            if row is None:
                raise KeyError(post_id)
            return self._post_record(row)

    def get_post_view(self, post_id: str, *, viewer_user_id: str | None = None) -> PlazaPostView:
        with self._session_factory() as session:
            post = session.get(PlazaPost, post_id)
            if post is None:
                raise KeyError(post_id)
            if session.get(VoiceProfile, post.voice_profile_id) is None:
                raise KeyError(post_id)
            [view] = self._assemble_views(session, [post], viewer_user_id)
            return view

    def is_published(self, voice_profile_id: str) -> bool:
        with self._session_factory() as session:
            return (
                session.scalar(
                    select(func.count(PlazaPost.id)).where(
                        PlazaPost.voice_profile_id == voice_profile_id
                    )
                )
                or 0
            ) > 0

    def is_published_ready(self, voice_profile_id: str) -> bool:
        with self._session_factory() as session:
            row = session.scalar(
                select(PlazaPost).where(PlazaPost.voice_profile_id == voice_profile_id)
            )
            if row is None:
                return False
            profile = session.get(VoiceProfile, voice_profile_id)
            return profile is not None and profile.status == "ready"

    def delete_post(self, post_id: str, actor_user_id: str) -> None:
        with self._session_factory() as session:
            row = session.get(PlazaPost, post_id)
            if row is None:
                raise KeyError(post_id)
            if row.author_user_id != actor_user_id:
                raise PlazaError("PLAZA_NOT_POST_AUTHOR", "仅作者可删除帖子")
            for model in (PlazaComment, PlazaLike, PlazaFavorite, Notification):
                for child in session.scalars(
                    select(model).where(model.post_id == post_id)
                ).all():
                    session.delete(child)
            session.delete(row)
            session.commit()

    def _assemble_views(
        self, session: Session, rows: list[PlazaPost], viewer_user_id: str | None
    ) -> list[PlazaPostView]:
        post_ids = [row.id for row in rows]
        like_counts = dict(
            session.execute(
                select(PlazaLike.post_id, func.count())
                .where(PlazaLike.post_id.in_(post_ids))
                .group_by(PlazaLike.post_id)
            ).all()
        )
        comment_counts = dict(
            session.execute(
                select(PlazaComment.post_id, func.count())
                .where(PlazaComment.post_id.in_(post_ids))
                .group_by(PlazaComment.post_id)
            ).all()
        )
        favorite_counts = dict(
            session.execute(
                select(PlazaFavorite.post_id, func.count())
                .where(PlazaFavorite.post_id.in_(post_ids))
                .group_by(PlazaFavorite.post_id)
            ).all()
        )
        viewer_likes: set[str] = set()
        viewer_favorites: set[str] = set()
        if viewer_user_id is not None:
            viewer_likes = set(
                session.scalars(
                    select(PlazaLike.post_id).where(
                        PlazaLike.post_id.in_(post_ids),
                        PlazaLike.user_id == viewer_user_id,
                    )
                ).all()
            )
            viewer_favorites = set(
                session.scalars(
                    select(PlazaFavorite.post_id).where(
                        PlazaFavorite.post_id.in_(post_ids),
                        PlazaFavorite.user_id == viewer_user_id,
                    )
                ).all()
            )
        references = session.scalars(
            select(VoiceReference)
            .where(VoiceReference.profile_id.in_([row.voice_profile_id for row in rows]))
            .order_by(VoiceReference.is_primary.desc(), VoiceReference.created_at.asc())
        ).all()
        refs_by_profile: dict[str, list[VoiceReference]] = {}
        for reference in references:
            refs_by_profile.setdefault(reference.profile_id, []).append(reference)
        usernames = {
            user.id: user.username
            for user in session.scalars(
                select(User).where(User.id.in_([row.author_user_id for row in rows]))
            ).all()
        }
        profiles_by_user = {
            profile.user_id: profile
            for profile in session.scalars(
                select(UserProfile).where(
                    UserProfile.user_id.in_([row.author_user_id for row in rows])
                )
            ).all()
        }
        views: list[PlazaPostView] = []
        for row in rows:
            profile = session.get(VoiceProfile, row.voice_profile_id)
            refs = refs_by_profile.get(row.voice_profile_id, [])
            author_profile = profiles_by_user.get(row.author_user_id)
            views.append(
                PlazaPostView(
                    post=self._post_record(row),
                    voice_name=public_voice_name(profile.id, profile.display_name),
                    voice_status=profile.status,
                    reference_count=len(refs),
                    reference_emotions=tuple(
                        dict.fromkeys(ref.emotion_label for ref in refs)
                    ),
                    author_display_name=(
                        author_profile.display_name
                        if author_profile is not None
                        else usernames.get(row.author_user_id, row.author_user_id)
                    ),
                    author_has_avatar=bool(
                        author_profile is not None and author_profile.avatar_path
                    ),
                    like_count=int(like_counts.get(row.id, 0)),
                    comment_count=int(comment_counts.get(row.id, 0)),
                    favorite_count=int(favorite_counts.get(row.id, 0)),
                    liked_by_viewer=row.id in viewer_likes,
                    favorited_by_viewer=row.id in viewer_favorites,
                )
            )
        return views

    def list_posts(
        self,
        *,
        viewer_user_id: str | None = None,
        mine: bool = False,
        favorited: bool = False,
        q: str | None = None,
        sort: str = "newest",
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[PlazaPostView], int]:
        with self._session_factory() as session:
            stmt = select(PlazaPost).join(
                VoiceProfile, PlazaPost.voice_profile_id == VoiceProfile.id
            )
            if mine:
                if viewer_user_id is None:
                    return [], 0
                stmt = stmt.where(PlazaPost.author_user_id == viewer_user_id)
            if favorited:
                if viewer_user_id is None:
                    return [], 0
                stmt = stmt.where(
                    PlazaPost.id.in_(
                        select(PlazaFavorite.post_id).where(
                            PlazaFavorite.user_id == viewer_user_id
                        )
                    )
                )
            if q:
                needle = f"%{q.strip()}%"
                stmt = stmt.where(
                    or_(
                        VoiceProfile.display_name.like(needle),
                        PlazaPost.description.like(needle),
                    )
                )
            rows = session.scalars(
                stmt.order_by(PlazaPost.created_at.desc(), PlazaPost.id.desc())
            ).all()
            if not rows:
                return [], 0
            views = self._assemble_views(session, rows, viewer_user_id)
            if sort == "likes":
                views.sort(
                    key=lambda view: (-view.like_count, view.post.created_at, view.post.id)
                )
            total = len(views)
            return views[offset : offset + limit], total

    def count_stats(self, user_id: str) -> tuple[int, int]:
        with self._session_factory() as session:
            published = int(
                session.scalar(
                    select(func.count(PlazaPost.id)).where(
                        PlazaPost.author_user_id == user_id
                    )
                )
                or 0
            )
            likes_received = int(
                session.scalar(
                    select(func.count(PlazaLike.user_id))
                    .select_from(PlazaLike)
                    .join(PlazaPost, PlazaLike.post_id == PlazaPost.id)
                    .where(PlazaPost.author_user_id == user_id)
                )
                or 0
            )
            return published, likes_received

    def _author_display(self, session: Session, user_id: str) -> tuple[str, bool]:
        username = session.scalar(select(User.username).where(User.id == user_id))
        profile = session.get(UserProfile, user_id)
        if profile is not None:
            return profile.display_name, bool(profile.avatar_path)
        return username or user_id, False

    def add_comment(self, post_id: str, author_user_id: str, content: str) -> PlazaCommentView:
        normalized = (content or "").strip()
        if not 1 <= len(normalized) <= MAX_COMMENT_LENGTH:
            raise PlazaError("PLAZA_COMMENT_INVALID", f"评论需在 1–{MAX_COMMENT_LENGTH} 字")
        if self.sensitive_filter.check(normalized).blocked:
            raise PlazaError("SENSITIVE_COMMENT_BLOCKED", "评论包含禁止发布的内容")
        with self._session_factory() as session:
            post = _require_post(session, post_id)
            row = PlazaComment(
                id=uuid4().hex,
                post_id=post.id,
                author_user_id=author_user_id,
                content=normalized,
            )
            session.add(row)
            session.commit()
            if self.notifications is not None:
                self.notifications.create(
                    post.author_user_id,
                    author_user_id,
                    type="comment",
                    post_id=post.id,
                    comment_id=row.id,
                )
            display, has_avatar = self._author_display(session, author_user_id)
            return PlazaCommentView(
                id=row.id,
                post_id=post.id,
                author_user_id=author_user_id,
                author_display_name=display,
                author_has_avatar=has_avatar,
                content=row.content,
                created_at=row.created_at,
            )

    def list_comments(
        self, post_id: str, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[PlazaCommentView], int]:
        with self._session_factory() as session:
            _require_post(session, post_id)
            total = int(
                session.scalar(
                    select(func.count(PlazaComment.id)).where(PlazaComment.post_id == post_id)
                )
                or 0
            )
            rows = session.scalars(
                select(PlazaComment)
                .where(PlazaComment.post_id == post_id)
                .order_by(PlazaComment.created_at.asc(), PlazaComment.id.asc())
                .limit(limit)
                .offset(offset)
            ).all()
            views = []
            for row in rows:
                display, has_avatar = self._author_display(session, row.author_user_id)
                views.append(
                    PlazaCommentView(
                        id=row.id,
                        post_id=post_id,
                        author_user_id=row.author_user_id,
                        author_display_name=display,
                        author_has_avatar=has_avatar,
                        content=row.content,
                        created_at=row.created_at,
                    )
                )
            return views, total

    def like(self, post_id: str, user_id: str) -> None:
        with self._session_factory() as session:
            post = _require_post(session, post_id)
            existing = session.get(PlazaLike, {"post_id": post_id, "user_id": user_id})
            if existing is not None:
                raise PlazaError("PLAZA_ALREADY_LIKED", "已点赞过该帖子")
            session.add(PlazaLike(post_id=post_id, user_id=user_id))
            session.commit()
        if self.notifications is not None:
            self.notifications.create(post.author_user_id, user_id, type="like", post_id=post_id)

    def unlike(self, post_id: str, user_id: str) -> None:
        with self._session_factory() as session:
            _require_post(session, post_id)
            existing = session.get(PlazaLike, {"post_id": post_id, "user_id": user_id})
            if existing is not None:
                session.delete(existing)
                session.commit()

    def favorite(self, post_id: str, user_id: str) -> None:
        with self._session_factory() as session:
            _require_post(session, post_id)
            existing = session.get(PlazaFavorite, {"post_id": post_id, "user_id": user_id})
            if existing is not None:
                raise PlazaError("PLAZA_ALREADY_FAVORITED", "已收藏过该帖子")
            session.add(PlazaFavorite(post_id=post_id, user_id=user_id))
            session.commit()

    def unfavorite(self, post_id: str, user_id: str) -> None:
        with self._session_factory() as session:
            _require_post(session, post_id)
            existing = session.get(PlazaFavorite, {"post_id": post_id, "user_id": user_id})
            if existing is not None:
                session.delete(existing)
                session.commit()

    def my_comments(
        self, user_id: str, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[PlazaCommentView], int]:
        with self._session_factory() as session:
            total = int(
                session.scalar(
                    select(func.count(PlazaComment.id)).where(
                        PlazaComment.author_user_id == user_id
                    )
                )
                or 0
            )
            rows = session.scalars(
                select(PlazaComment)
                .where(PlazaComment.author_user_id == user_id)
                .order_by(PlazaComment.created_at.desc(), PlazaComment.id.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            views = []
            for row in rows:
                display, has_avatar = self._author_display(session, row.author_user_id)
                post = session.get(PlazaPost, row.post_id)
                voice_name = None
                if post is not None:
                    profile = session.get(VoiceProfile, post.voice_profile_id)
                    if profile is not None:
                        voice_name = public_voice_name(profile.id, profile.display_name)
                views.append(
                    PlazaCommentView(
                        id=row.id,
                        post_id=row.post_id,
                        author_user_id=row.author_user_id,
                        author_display_name=display,
                        author_has_avatar=has_avatar,
                        content=row.content,
                        created_at=row.created_at,
                        voice_name=voice_name,
                    )
                )
            return views, total
