"""Initialize the one local administrator without placing a password in argv."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.db.session import create_database_engine  # noqa: E402
from backend.app.services.auth import AuthError, AuthService  # noqa: E402


def _database_url(storage_root: Path) -> str:
    database_path = (storage_root / "app.db").resolve()
    return f"sqlite+pysqlite:///{database_path.as_posix()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="初始化本地 timbre-share 管理员账号")
    parser.add_argument("--username", help="管理员用户名；密码始终通过隐藏式提示输入")
    args = parser.parse_args(argv)

    settings = get_settings(ROOT / "config" / "app.yaml")
    storage_root = settings.storage.root
    if not storage_root.is_absolute():
        storage_root = ROOT / storage_root
    storage_root.mkdir(parents=True, exist_ok=True)

    username = (args.username or input("管理员用户名: ")).strip()
    password = getpass.getpass("管理员密码（至少 8 位）: ")
    confirmation = getpass.getpass("再次输入管理员密码: ")
    if password != confirmation:
        print("两次密码不一致。", file=sys.stderr)
        return 2

    try:
        admin = AuthService(engine=create_database_engine(_database_url(storage_root))).bootstrap_admin(
            username, password
        )
    except AuthError as exc:
        print(f"管理员初始化失败：{exc}", file=sys.stderr)
        return 1

    print(f"管理员已初始化：{admin.username}（本地 SQLite）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
