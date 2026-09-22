from pathlib import Path

from sqlalchemy import Engine, create_engine, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import VoiceProfile


def create_database_engine(database_url: str) -> Engine:
    if database_url in {"sqlite://", "sqlite:///:memory:", "sqlite+pysqlite:///:memory:"}:
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    if database_url.startswith("sqlite:"):
        Path(database_url.removeprefix("sqlite+pysqlite:///").removeprefix("sqlite:///"))
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
    return create_engine(database_url)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as connection:
        columns_by_table = {
            table: {
                row[1]
                for row in connection.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            }
            for table in ("datasets", "audio_assets", "voice_profiles", "jobs", "syntheses", "voice_references")
        }
        required_columns = {
            "datasets": {
                "owner_user_id": "VARCHAR(64)",
                "authorization_confirmed_at": "DATETIME",
                "effective_seconds": "FLOAT",
            },
            "audio_assets": {"owner_user_id": "VARCHAR(64)"},
            "voice_profiles": {
                "dataset_id": "VARCHAR(64)",
                "owner_user_id": "VARCHAR(64)",
                "mode": "VARCHAR(32)",
                "reference_asset_id": "VARCHAR(64)",
                "reference_segment_id": "VARCHAR(64)",
                "prompt_text": "TEXT",
                "prompt_language": "VARCHAR(8)",
                "base_model_id": "VARCHAR(64)",
                "display_name": "VARCHAR(64)",
            },
            "jobs": {"owner_user_id": "VARCHAR(64)", "progress_message": "TEXT"},
            "syntheses": {"owner_user_id": "VARCHAR(64)"},
            "voice_references": {"reference_name": "VARCHAR(64)"},
        }
        for table, columns in required_columns.items():
            for column, definition in columns.items():
                if column not in columns_by_table[table]:
                    connection.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                    )
        for table in required_columns:
            connection.exec_driver_sql(
                f"CREATE INDEX IF NOT EXISTS ix_{table}_owner_user_id ON {table} (owner_user_id)"
            )
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_voice_profiles_dataset_id ON voice_profiles (dataset_id)"
        )
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_voice_references_profile_id_owner_user_id "
            "ON voice_references (profile_id, owner_user_id)"
        )
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_voice_references_asset_id "
            "ON voice_references (asset_id)"
        )
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_voice_references_segment_id "
            "ON voice_references (segment_id)"
        )
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_voice_references_one_primary "
            "ON voice_references (profile_id) WHERE is_primary = 1"
        )


def migrate_voice_base_id(engine: Engine, old_id: str, new_id: str) -> int:
    """Move zero-shot profiles from one immutable base-model identity to another."""
    if not old_id or not new_id:
        raise ValueError("voice base model ids must not be empty")
    if old_id == new_id:
        return 0
    with engine.begin() as connection:
        result = connection.execute(
            update(VoiceProfile)
            .where(VoiceProfile.mode == "zero_shot")
            .where(VoiceProfile.base_model_id == old_id)
            .values(base_model_id=new_id)
        )
    return int(result.rowcount or 0)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
