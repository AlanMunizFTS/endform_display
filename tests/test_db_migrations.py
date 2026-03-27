import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import db


class _FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self._results = []

    def execute(self, query, params=None):
        normalized = " ".join(str(query).split())
        self.connection.executed.append((normalized, params))

        if normalized.startswith("SELECT pg_advisory_lock"):
            self._results = [(True,)]
            return

        if normalized.startswith("SELECT pg_advisory_unlock"):
            self._results = [(True,)]
            return

        if (
            normalized.startswith("CREATE TABLE IF NOT EXISTS schema_migrations")
            or normalized.startswith("SELECT filename FROM schema_migrations")
        ):
            if normalized.startswith("SELECT filename FROM schema_migrations"):
                self._results = [
                    (name,) for name in sorted(self.connection.applied_migrations)
                ]
            else:
                self._results = []
            return

        if normalized.startswith("INSERT INTO schema_migrations"):
            self.connection.applied_migrations.add(params[0])
            self._results = []
            return

        if "RAISE_MIGRATION_ERROR" in normalized:
            raise RuntimeError("migration failed")

        self.connection.applied_sql.append(normalized)
        self._results = []

    def fetchall(self):
        return list(self._results)

    def close(self):
        return None


class _FakeConnection:
    def __init__(self):
        self.executed = []
        self.applied_sql = []
        self.applied_migrations = set()
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = 0

    def cursor(self, *args, **kwargs):
        return _FakeCursor(self)

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


class _FakePool:
    def __init__(self, connection):
        self.connection = connection
        self.closeall_called = False
        self.getconn_count = 0
        self.putconn_count = 0

    def getconn(self):
        self.getconn_count += 1
        return self.connection

    def putconn(self, connection):
        self.putconn_count += 1

    def closeall(self):
        self.closeall_called = True


class TestDbMigrations(unittest.TestCase):
    def _build_db_settings(self):
        return {
            "host": "localhost",
            "port": 5432,
            "database": "endform",
            "user": "postgres",
            "password": "secret",
        }

    def _build_client(self, connection):
        pool = _FakePool(connection)
        pool_factory = patch(
            "db.psycopg2.pool.SimpleConnectionPool",
            return_value=pool,
        )
        settings_factory = patch("db.get_db_settings", return_value=self._build_db_settings())
        return pool, pool_factory, settings_factory

    def test_get_db_connection_applies_pending_migrations_in_order(self):
        connection = _FakeConnection()
        pool, pool_factory, settings_factory = self._build_client(connection)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "001_first.sql").write_text("CREATE TABLE first_table (id INT);", encoding="utf-8")
            Path(tmpdir, "002_second.sql").write_text("ALTER TABLE first_table ADD COLUMN name TEXT;", encoding="utf-8")

            with pool_factory, settings_factory, patch("db.MIGRATIONS_DIR", Path(tmpdir)):
                client = db.get_db_connection()

        self.assertEqual(
            connection.applied_sql,
            [
                "CREATE TABLE first_table (id INT);",
                "ALTER TABLE first_table ADD COLUMN name TEXT;",
            ],
        )
        self.assertEqual(
            connection.applied_migrations,
            {"001_first.sql", "002_second.sql"},
        )
        self.assertGreaterEqual(connection.commit_count, 4)
        self.assertGreaterEqual(pool.getconn_count, 2)
        self.assertGreaterEqual(pool.putconn_count, 2)
        client.close()

    def test_get_db_connection_accepts_utf8_bom_migrations(self):
        connection = _FakeConnection()
        pool, pool_factory, settings_factory = self._build_client(connection)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "001_first.sql").write_text(
                "\ufeff-- migration with bom\nCREATE TABLE first_table (id INT);",
                encoding="utf-8",
            )

            with pool_factory, settings_factory, patch("db.MIGRATIONS_DIR", Path(tmpdir)):
                client = db.get_db_connection()

        self.assertEqual(
            connection.applied_sql,
            ["-- migration with bom CREATE TABLE first_table (id INT);"],
        )
        self.assertEqual(connection.applied_migrations, {"001_first.sql"})
        client.close()

    def test_get_db_connection_skips_already_applied_migrations(self):
        connection = _FakeConnection()
        connection.applied_migrations.update({"001_first.sql", "002_second.sql"})
        pool, pool_factory, settings_factory = self._build_client(connection)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "001_first.sql").write_text("CREATE TABLE first_table (id INT);", encoding="utf-8")
            Path(tmpdir, "002_second.sql").write_text("ALTER TABLE first_table ADD COLUMN name TEXT;", encoding="utf-8")

            with pool_factory, settings_factory, patch("db.MIGRATIONS_DIR", Path(tmpdir)):
                client = db.get_db_connection()

        self.assertEqual(connection.applied_sql, [])
        self.assertEqual(
            connection.applied_migrations,
            {"001_first.sql", "002_second.sql"},
        )
        client.close()

    def test_get_db_connection_rolls_back_failed_migration_and_closes_pool(self):
        connection = _FakeConnection()
        pool, pool_factory, settings_factory = self._build_client(connection)

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "001_first.sql").write_text("CREATE TABLE first_table (id INT);", encoding="utf-8")
            Path(tmpdir, "002_broken.sql").write_text("RAISE_MIGRATION_ERROR", encoding="utf-8")

            with pool_factory, settings_factory, patch("db.MIGRATIONS_DIR", Path(tmpdir)):
                with self.assertRaises(RuntimeError):
                    db.get_db_connection()

        self.assertEqual(connection.applied_sql, ["CREATE TABLE first_table (id INT);"])
        self.assertEqual(connection.applied_migrations, {"001_first.sql"})
        self.assertGreaterEqual(connection.rollback_count, 1)
        self.assertTrue(pool.closeall_called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
