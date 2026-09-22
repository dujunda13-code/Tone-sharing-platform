"""NotificationStore: recipient-scoped notifications with enriched views."""

import time

from sqlalchemy import create_engine

from backend.app.db.models import User, VoiceProfile
from backend.app.db.session import init_db, session_factory

from backend.app.services.notifications import NotificationStore


def _seed_user(engine, user_id: str, username: str, display_name: str | None = None):
    with session_factory(engine)() as session:
        session.add(
            User(
                id=user_id,
                username=username,
                password_hash="x",
                password_salt="s",
                password_params_json="{}",
            )
        )
        if display_name:
            session.add(
                VoiceProfile(  # placeholder profile so views resolve voice names
                    id=f"vp-{user_id}",
                    owner_user_id=user_id,
                    dataset_id=f"ds-{user_id}",
                    status="ready",
                    mode="zero_shot",
                    display_name=f"音色-{display_name}",
                )
            )
        session.commit()


def _engine(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'notify.db').as_posix()}")
    init_db(engine)
    return engine


def test_create_skips_self_notifications_and_validates_type(tmp_path):
    engine = _engine(tmp_path)
    store = NotificationStore(engine=engine)

    assert store.create("u1", "u1", type="like", post_id="p1") is None
    created = store.create("u1", "u2", type="like", post_id="p1")
    assert created is not None and created.type == "like"

    try:
        store.create("u1", "u2", type="follow", post_id="p1")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown notification type must be rejected")


def test_list_views_pagination_order_and_enrichment(tmp_path):
    engine = _engine(tmp_path)
    _seed_user(engine, "alice", "alice", display_name="Alice")
    _seed_user(engine, "bob", "bob")
    store = NotificationStore(engine=engine)
    with session_factory(engine)() as session:
        session.add(
            VoiceProfile(
                id="vp-post",
                owner_user_id="alice",
                dataset_id="ds-post",
                status="ready",
                mode="zero_shot",
                display_name="被赞的音色",
            )
        )
        session.commit()
    from backend.app.db.models import PlazaComment, PlazaPost

    with session_factory(engine)() as session:
        session.add(PlazaPost(id="p1", author_user_id="alice", voice_profile_id="vp-post"))
        session.add(
            PlazaComment(
                id="c1",
                post_id="p1",
                author_user_id="bob",
                content="这条评论超过五十个字吗" + "行" * 40,
            )
        )
        session.commit()
    store.create("alice", "bob", type="like", post_id="p1")
    time.sleep(0.002)
    store.create("alice", "bob", type="comment", post_id="p1", comment_id="c1")

    views, total = store.list_views("alice", limit=1, offset=0)

    assert total == 2
    assert len(views) == 1
    assert views[0].type == "comment"  # newest first
    assert views[0].voice_name == "被赞的音色"
    assert views[0].comment_excerpt is not None and len(views[0].comment_excerpt) <= 50


def test_unread_count_and_mark_read_scoping(tmp_path):
    engine = _engine(tmp_path)
    store = NotificationStore(engine=engine)
    store.create("alice", "bob", type="like", post_id="p1")
    store.create("alice", "carol", type="like", post_id="p1")
    store.create("bob", "alice", type="like", post_id="p2")

    assert store.unread_count("alice") == 2
    assert store.unread_count("bob") == 1

    first = store.list_views("alice", limit=1)[0][0]
    updated = store.mark_read("alice", notification_ids=[first.id])

    assert updated == 1
    assert store.unread_count("alice") == 1
    # Bob cannot mark Alice's notification.
    assert store.mark_read("bob", notification_ids=[first.id]) == 0
    assert store.mark_read("alice", all=True) >= 1
    assert store.unread_count("alice") == 0


def test_delete_for_post_removes_only_that_post(tmp_path):
    engine = _engine(tmp_path)
    store = NotificationStore(engine=engine)
    store.create("alice", "bob", type="like", post_id="p1")
    store.create("alice", "bob", type="like", post_id="p2")

    store.delete_for_post("p1")

    assert store.unread_count("alice") == 1
