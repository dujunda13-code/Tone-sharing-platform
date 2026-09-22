"""PlazaStore post domain: publish gating, listing, cascade delete, interactions."""

import time
from uuid import uuid4

from sqlalchemy import create_engine, func, select

from backend.app.db.models import (
    Dataset,
    Notification,
    PlazaComment,
    PlazaFavorite,
    PlazaLike,
    User,
    VoiceProfile,
    utc_now,
)
from backend.app.db.session import init_db, session_factory

from backend.app.services.notifications import NotificationStore
from backend.app.services.plaza import PlazaError, PlazaStore
from backend.app.services.sensitive_filter import SensitiveFilter


def _engine(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'plaza.db').as_posix()}")
    init_db(engine)
    return engine


def _store(engine) -> PlazaStore:
    return PlazaStore(
        engine=engine,
        sensitive_filter=SensitiveFilter([]),
        notifications=NotificationStore(engine=engine),
    )


def _interactive_store(engine) -> PlazaStore:
    return PlazaStore(
        engine=engine,
        sensitive_filter=SensitiveFilter(["违禁词"]),
        notifications=NotificationStore(engine=engine),
    )


def _seed_user(engine, user_id: str):
    with session_factory(engine)() as session:
        session.add(
            User(
                id=user_id,
                username=user_id,
                password_hash="x",
                password_salt="s",
                password_params_json="{}",
            )
        )
        session.commit()


def _seed_profile(engine, profile_id: str, owner: str, *, status="ready", name="测试音色"):
    with session_factory(engine)() as session:
        session.add(
            Dataset(
                id=f"ds-{profile_id}",
                owner_user_id=owner,
                authorization_confirmed_at=utc_now(),
                effective_seconds=6.0,
                status="ready_for_profile",
            )
        )
        session.add(
            VoiceProfile(
                id=profile_id,
                owner_user_id=owner,
                dataset_id=f"ds-{profile_id}",
                status=status,
                mode="zero_shot",
                base_model_id="gpt-sovits-v2proplus-official",
                display_name=name,
            )
        )
        session.commit()


def _seed_rows(engine, rows):
    with session_factory(engine)() as session:
        for row in rows:
            session.add(row)
        session.commit()


def test_publish_gates_owner_ready_and_uniqueness(tmp_path):
    engine = _engine(tmp_path)
    store = _store(engine)
    _seed_user(engine, "alice")
    _seed_user(engine, "bob")
    _seed_profile(engine, "vp1", "alice")
    _seed_profile(engine, "vp-notready", "alice", status="created")

    post = store.publish("alice", voice_profile_id="vp1", description=" 我的音色 ")
    assert post.description == "我的音色"
    assert store.is_published("vp1")
    assert store.is_published_ready("vp1")

    try:
        store.publish("bob", voice_profile_id="vp1")
    except KeyError:
        pass
    else:
        raise AssertionError("foreign profile must 404")

    try:
        store.publish("alice", voice_profile_id="vp-notready")
    except PlazaError as exc:
        assert exc.code == "PLAZA_VOICE_PROFILE_NOT_READY"
    else:
        raise AssertionError("non-ready profile must be rejected")

    try:
        store.publish("alice", voice_profile_id="vp1")
    except PlazaError as exc:
        assert exc.code == "PLAZA_POST_ALREADY_PUBLISHED"
    else:
        raise AssertionError("duplicate publish must be rejected")

    try:
        store.publish("alice", voice_profile_id="missing")
    except KeyError:
        pass
    else:
        raise AssertionError("missing profile must 404")

    try:
        store.publish("alice", voice_profile_id="vp1", description="x" * 501)
    except PlazaError as exc:
        assert exc.code == "PLAZA_DESCRIPTION_TOO_LONG"
    else:
        raise AssertionError("overlong description must be rejected")


def test_list_posts_counts_filters_search_and_sort(tmp_path):
    engine = _engine(tmp_path)
    store = _store(engine)
    _seed_user(engine, "alice")
    _seed_user(engine, "bob")
    _seed_user(engine, "carol")
    _seed_profile(engine, "vp1", "alice", name="清亮女声")
    _seed_profile(engine, "vp2", "bob", name="低沉男声")
    p1 = store.publish("alice", voice_profile_id="vp1", description="温柔故事")
    p2 = store.publish("bob", voice_profile_id="vp2")

    _seed_rows(
        engine,
        [
            PlazaLike(post_id=p2.id, user_id="alice"),
            PlazaLike(post_id=p2.id, user_id="carol"),
            PlazaLike(post_id=p1.id, user_id="bob"),
            PlazaFavorite(post_id=p2.id, user_id="alice"),
        ],
    )

    views, total = store.list_posts(viewer_user_id="alice")
    assert total == 2
    by_id = {v.post.id: v for v in views}
    assert by_id[p2.id].like_count == 2 and by_id[p2.id].liked_by_viewer
    assert by_id[p2.id].favorited_by_viewer
    assert by_id[p1.id].like_count == 1 and not by_id[p1.id].liked_by_viewer
    assert by_id[p1.id].voice_name == "清亮女声"
    assert by_id[p1.id].author_display_name == "alice"

    mine, _ = store.list_posts(viewer_user_id="alice", mine=True)
    assert [v.post.id for v in mine] == [p1.id]

    favs, _ = store.list_posts(viewer_user_id="alice", favorited=True)
    assert [v.post.id for v in favs] == [p2.id]

    hits, _ = store.list_posts(q="清亮")
    assert [v.post.id for v in hits] == [p1.id]
    hits, _ = store.list_posts(q="温柔故事")
    assert [v.post.id for v in hits] == [p1.id]

    likes_sorted, _ = store.list_posts(sort="likes")
    assert likes_sorted[0].post.id == p2.id

    page, total = store.list_posts(viewer_user_id="alice", limit=1, offset=1)
    assert total == 2 and len(page) == 1


def test_delete_post_only_author_and_cascades(tmp_path):
    engine = _engine(tmp_path)
    store = _store(engine)
    _seed_user(engine, "alice")
    _seed_user(engine, "bob")
    _seed_profile(engine, "vp1", "alice")
    p1 = store.publish("alice", voice_profile_id="vp1")
    _seed_rows(
        engine,
        [
            PlazaLike(post_id=p1.id, user_id="bob"),
            PlazaFavorite(post_id=p1.id, user_id="bob"),
            PlazaComment(
                id=uuid4().hex, post_id=p1.id, author_user_id="bob", content="很好听"
            ),
            Notification(
                id=uuid4().hex,
                recipient_user_id="alice",
                actor_user_id="bob",
                type="like",
                post_id=p1.id,
            ),
        ],
    )

    try:
        store.delete_post(p1.id, "bob")
    except PlazaError as exc:
        assert exc.code == "PLAZA_NOT_POST_AUTHOR"
    else:
        raise AssertionError("non-author delete must be rejected")

    store.delete_post(p1.id, "alice")

    with session_factory(engine)() as session:
        for model in (PlazaComment, PlazaLike, PlazaFavorite, Notification):
            assert session.scalar(select(func.count()).select_from(model)) == 0
    try:
        store.get_post(p1.id)
    except KeyError:
        pass
    else:
        raise AssertionError("deleted post must be gone")

    try:
        store.delete_post("missing", "alice")
    except KeyError:
        pass
    else:
        raise AssertionError("missing post must 404")


def test_count_stats(tmp_path):
    engine = _engine(tmp_path)
    store = _store(engine)
    _seed_user(engine, "alice")
    _seed_user(engine, "bob")
    _seed_profile(engine, "vp1", "alice")
    _seed_profile(engine, "vp2", "alice")
    p1 = store.publish("alice", voice_profile_id="vp1")
    store.publish("alice", voice_profile_id="vp2")
    _seed_rows(engine, [PlazaLike(post_id=p1.id, user_id="bob")])

    assert store.count_stats("alice") == (2, 1)
    assert store.count_stats("bob") == (0, 0)


def test_add_comment_gates_and_notifies(tmp_path):
    engine = _engine(tmp_path)
    store = _interactive_store(engine)
    _seed_user(engine, "alice")
    _seed_user(engine, "bob")
    _seed_profile(engine, "vp1", "alice")
    p1 = store.publish("alice", voice_profile_id="vp1")

    comment = store.add_comment(p1.id, "bob", " 很好听")
    assert comment.content == "很好听"
    assert comment.author_display_name == "bob"
    assert store.notifications.unread_count("alice") == 1

    time.sleep(0.002)
    store.add_comment(p1.id, "alice", "自评不通知")
    assert store.notifications.unread_count("alice") == 1  # self comment: no notification

    for bad in ["", "   ", "x" * 501]:
        try:
            store.add_comment(p1.id, "bob", bad)
        except PlazaError as exc:
            assert exc.code == "PLAZA_COMMENT_INVALID"
        else:
            raise AssertionError(f"must reject {bad!r}")

    try:
        store.add_comment(p1.id, "bob", "这句话里有违禁词")
    except PlazaError as exc:
        assert exc.code == "SENSITIVE_COMMENT_BLOCKED"
    else:
        raise AssertionError("sensitive comment must be blocked")

    try:
        store.add_comment("missing", "bob", "内容")
    except KeyError:
        pass
    else:
        raise AssertionError("missing post must 404")

    comments, total = store.list_comments(p1.id)
    assert total == 2 and comments[0].content == "很好听"


def test_like_and_favorite_lifecycle_with_notifications(tmp_path):
    engine = _engine(tmp_path)
    store = _interactive_store(engine)
    _seed_user(engine, "alice")
    _seed_user(engine, "bob")
    _seed_profile(engine, "vp1", "alice")
    p1 = store.publish("alice", voice_profile_id="vp1")

    store.like(p1.id, "bob")
    assert store.notifications.unread_count("alice") == 1
    store.like(p1.id, "alice")  # self-like: no notification
    assert store.notifications.unread_count("alice") == 1

    try:
        store.like(p1.id, "bob")
    except PlazaError as exc:
        assert exc.code == "PLAZA_ALREADY_LIKED"
    else:
        raise AssertionError("duplicate like must be rejected")

    store.unlike(p1.id, "bob")
    store.unlike(p1.id, "bob")  # idempotent
    views, _ = store.list_posts(viewer_user_id="bob")
    assert views[0].like_count == 1  # alice's self-like still there

    store.favorite(p1.id, "bob")
    try:
        store.favorite(p1.id, "bob")
    except PlazaError as exc:
        assert exc.code == "PLAZA_ALREADY_FAVORITED"
    else:
        raise AssertionError("duplicate favorite must be rejected")
    store.unfavorite(p1.id, "bob")
    store.unfavorite(p1.id, "bob")  # idempotent

    try:
        store.like("missing", "bob")
    except KeyError:
        pass
    else:
        raise AssertionError("missing post must 404")


def test_my_comments_includes_voice_name(tmp_path):
    engine = _engine(tmp_path)
    store = _interactive_store(engine)
    _seed_user(engine, "alice")
    _seed_user(engine, "bob")
    _seed_profile(engine, "vp1", "alice", name="广场音色A")
    p1 = store.publish("alice", voice_profile_id="vp1")
    store.add_comment(p1.id, "bob", "第一条")

    mine, total = store.my_comments("bob")

    assert total == 1
    assert mine[0].voice_name == "广场音色A"
    assert mine[0].content == "第一条"
