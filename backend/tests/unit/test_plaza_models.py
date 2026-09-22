"""Social-domain schema: six new tables created by init_db (Voice Plaza)."""

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from backend.app.db.session import init_db


def _engine(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'social.db').as_posix()}")
    init_db(engine)
    return engine


def test_init_db_creates_all_social_tables(tmp_path):
    engine = _engine(tmp_path)
    with engine.connect() as connection:
        names = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert {
        "plaza_posts",
        "plaza_comments",
        "plaza_likes",
        "plaza_favorites",
        "notifications",
        "user_profiles",
    } <= names


def test_plaza_posts_voice_profile_id_is_unique(tmp_path):
    engine = _engine(tmp_path)
    insert = (
        "INSERT INTO plaza_posts (id, author_user_id, voice_profile_id, created_at) "
        "VALUES ('{id}', '{author}', '{profile}', '2026-09-17 00:00:00+00:00')"
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(insert.format(id="p1", author="u1", profile="vp1"))
        try:
            connection.exec_driver_sql(insert.format(id="p2", author="u2", profile="vp1"))
        except IntegrityError:
            pass
        else:
            raise AssertionError("duplicate voice_profile_id must be rejected")


def test_plaza_likes_composite_primary_key_rejects_duplicate(tmp_path):
    engine = _engine(tmp_path)
    insert = (
        "INSERT INTO plaza_likes (post_id, user_id, created_at) "
        "VALUES ('p1', 'u1', '2026-09-17 00:00:00+00:00')"
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(insert)
        try:
            connection.exec_driver_sql(insert)
        except IntegrityError:
            pass
        else:
            raise AssertionError("duplicate (post_id, user_id) must be rejected")
