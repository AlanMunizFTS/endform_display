import re
import select
import socketserver
import threading
from contextlib import contextmanager
from pathlib import Path

import paramiko
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from settings import get_db_settings
from utilities.log import get_logger

logger = get_logger()

SCHEMA_MIGRATIONS_TABLE = "schema_migrations"
SCHEMA_MIGRATIONS_LOCK_KEY = 420731145922714061
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
READ_ONLY_SQL_PREFIXES = ("select", "show", "explain")
MUTATING_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|truncate|alter|drop|create|merge|grant|revoke|call)\b",
    re.IGNORECASE,
)


def _strip_leading_sql_comments(query):
    sql = str(query or "").lstrip()
    while True:
        if sql.startswith("--"):
            newline_idx = sql.find("\n")
            if newline_idx < 0:
                return ""
            sql = sql[newline_idx + 1 :].lstrip()
            continue
        if sql.startswith("/*"):
            end_idx = sql.find("*/")
            if end_idx < 0:
                return ""
            sql = sql[end_idx + 2 :].lstrip()
            continue
        return sql


def _is_read_only_sql(query):
    sql = _strip_leading_sql_comments(query).lower()
    if not sql:
        return False
    if ";" in sql.rstrip("; \t\r\n"):
        return False
    if MUTATING_SQL_PATTERN.search(sql):
        return False
    first_token = re.split(r"\s+", sql, maxsplit=1)[0]
    return first_token in READ_ONLY_SQL_PREFIXES


class _SSHTunnelRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        tunnel = getattr(self.server, "tunnel", None)
        if tunnel is None or tunnel.ssh_client is None:
            return

        transport = tunnel.ssh_client.get_transport()
        if transport is None or not transport.is_active():
            return

        channel = None
        try:
            channel = transport.open_channel(
                "direct-tcpip",
                (tunnel.remote_host, tunnel.remote_port),
                self.request.getpeername(),
            )
            if channel is None:
                raise RuntimeError("SSH tunnel channel could not be opened")

            while not tunnel.stop_event.is_set():
                readable, _, _ = select.select([self.request, channel], [], [], 0.5)
                if not readable:
                    continue

                for stream in readable:
                    data = stream.recv(4096)
                    if not data:
                        return

                    if stream is self.request:
                        channel.sendall(data)
                    else:
                        self.request.sendall(data)
        finally:
            try:
                if channel is not None:
                    channel.close()
            except Exception:
                pass
            try:
                self.request.close()
            except Exception:
                pass


class _SSHTunnelServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class SSHTunnelForwarder:
    def __init__(
        self,
        ssh_host,
        ssh_port,
        ssh_username,
        ssh_password,
        remote_host,
        remote_port,
        local_host="127.0.0.1",
        local_port=0,
    ):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_host = local_host
        self.local_port = local_port
        self.ssh_client = None
        self.server = None
        self.server_thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.server is not None:
            return

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(
            hostname=self.ssh_host,
            port=self.ssh_port,
            username=self.ssh_username,
            password=self.ssh_password,
            timeout=10,
        )

        self.server = _SSHTunnelServer(
            (self.local_host, self.local_port),
            _SSHTunnelRequestHandler,
        )
        self.server.tunnel = self
        self.local_port = self.server.server_address[1]

        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="ssh-tunnel-forwarder",
            daemon=True,
        )
        self.server_thread.start()

    def close(self):
        self.stop_event.set()

        if self.server is not None:
            try:
                self.server.shutdown()
            except Exception:
                pass
            try:
                self.server.server_close()
            except Exception:
                pass
            self.server = None

        if self.server_thread is not None:
            try:
                self.server_thread.join(timeout=2)
            except Exception:
                pass
            self.server_thread = None

        if self.ssh_client is not None:
            try:
                self.ssh_client.close()
            except Exception:
                pass
            self.ssh_client = None


class PostgresDB:
    def __init__(self, host, port, database, user, password, options=None):
        """Initialize PostgreSQL connection with connection pool"""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.options = options
        
        try:
            logger.info(
                f"[DB] Connecting to PostgreSQL at {host}:{port}, database={database}, user={user}",
            )
            connection_kwargs = {
                "host": host,
                "port": port,
                "database": database,
                "user": user,
                "password": password,
            }
            if options:
                connection_kwargs["options"] = options
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                10,
                **connection_kwargs,
            )
            # Validate a live connection early so startup errors are explicit.
            connection = self.connection_pool.getconn()
            self.connection_pool.putconn(connection)
            logger.info("[DB] Connection successful")
        except Exception as e:
            logger.error(f"[DB] Connection error: {e}")
            raise
    
    @contextmanager
    def get_cursor(self):
        """Get cursor with auto-commit/rollback"""
        connection = self.connection_pool.getconn()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise e
        finally:
            cursor.close()
            self.connection_pool.putconn(connection)
    
    def execute(self, query, data=None):
        """
        Execute any query (INSERT, UPDATE, DELETE)
        
        Args:
            query: SQL query with %s placeholders
            data: Tuple or list with values
            
        Returns:
            Number of affected rows
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, data)
                return cursor.rowcount
        except Exception as e:
            logger.error(f"DB execute error: {e}")
            raise
    
    def fetch(self, query, data=None):
        """
        Execute SELECT query and return results
        
        Args:
            query: SQL query with %s placeholders
            data: Tuple or list with values
            
        Returns:
            List of dictionaries
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, data)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"DB fetch error: {e}")
            raise

    def truncate_app_tables(self):
        """
        Truncate only the tables managed by this application.

        Returns:
            Number of truncated tables
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    "TRUNCATE TABLE piece_result, classified_images, img_results, model_results, "
                    "remote_model_results_pending, remote_sync_state "
                    "RESTART IDENTITY CASCADE"
                )
                return 6
        except Exception as e:
            logger.error(f"DB truncate_app_tables error: {e}")
            raise
    
    def close(self):
        """Close all connections"""
        if self.connection_pool:
            self.connection_pool.closeall()

    def insert_img(self, img_name):
        """Insert image name into images table"""
        query = "INSERT INTO img_results (name) VALUES (%s)"
        return self.execute(query, (img_name,))

    def apply_local_migrations(self, migrations_dir=None):
        """Apply pending local SQL migrations tracked in schema_migrations."""
        migrations_path = Path(migrations_dir) if migrations_dir else MIGRATIONS_DIR
        migration_files = sorted(
            path for path in migrations_path.glob("*.sql") if path.is_file()
        )

        if not migration_files:
            logger.info(
                f"[DB] No migrations found in {migrations_path}",
                allow_repeat=True,
            )
            return []

        connection = self.connection_pool.getconn()
        cursor = None
        applied_now = []
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT pg_advisory_lock(%s)",
                (SCHEMA_MIGRATIONS_LOCK_KEY,),
            )
            connection.commit()

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA_MIGRATIONS_TABLE} (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

            cursor.execute(
                f"SELECT filename FROM {SCHEMA_MIGRATIONS_TABLE} ORDER BY filename"
            )
            applied_rows = cursor.fetchall()
            applied_filenames = {row[0] for row in applied_rows}
            pending_files = [
                path for path in migration_files if path.name not in applied_filenames
            ]

            if not pending_files:
                logger.debug("[DB] Schema already up to date", allow_repeat=True)
                return []

            logger.info(
                f"[DB] Applying {len(pending_files)} pending migration(s)",
                allow_repeat=True,
            )
            for migration_file in pending_files:
                # Accept UTF-8 files with BOM so SQL comments at line 1 do not break parsing.
                migration_sql = migration_file.read_text(encoding="utf-8-sig").strip()
                logger.info(
                    f"[DB] Applying migration {migration_file.name}",
                    allow_repeat=True,
                )

                try:
                    if migration_sql:
                        cursor.execute(migration_sql)
                    else:
                        logger.warn(
                            f"[DB] Migration {migration_file.name} is empty; recording as applied",
                            allow_repeat=True,
                        )

                    cursor.execute(
                        f"INSERT INTO {SCHEMA_MIGRATIONS_TABLE} (filename) VALUES (%s)",
                        (migration_file.name,),
                    )
                    connection.commit()
                    applied_now.append(migration_file.name)
                except Exception as exc:
                    connection.rollback()
                    logger.error(
                        f"[DB] Migration {migration_file.name} failed: {exc}"
                    )
                    raise

            logger.info(
                "[DB] Applied migrations: " + ", ".join(applied_now),
                allow_repeat=True,
            )
            return applied_now
        except Exception:
            connection.rollback()
            raise
        finally:
            if cursor is not None:
                try:
                    connection.rollback()
                    cursor.execute(
                        "SELECT pg_advisory_unlock(%s)",
                        (SCHEMA_MIGRATIONS_LOCK_KEY,),
                    )
                    connection.commit()
                except Exception:
                    try:
                        connection.rollback()
                    except Exception:
                        pass
                try:
                    cursor.close()
                except Exception:
                    pass
            self.connection_pool.putconn(connection)


class SSHTunneledPostgresDB(PostgresDB):
    def __init__(self, tunnel, database, user, password, read_only=True):
        self.tunnel = tunnel
        self.read_only = bool(read_only)
        try:
            super().__init__(
                host="127.0.0.1",
                port=tunnel.local_port,
                database=database,
                user=user,
                password=password,
                options=(
                    "-c default_transaction_read_only=on"
                    if self.read_only
                    else None
                ),
            )
        except Exception:
            self.tunnel.close()
            raise

        if self.read_only:
            self._enable_read_only_transactions()

    def _enable_read_only_transactions(self):
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SET default_transaction_read_only = on")
        except Exception as exc:
            self.tunnel.close()
            logger.error(f"[REMOTE_DB] Failed to enable read-only mode: {exc}")
            raise

    def _validate_read_only_query(self, query):
        if self.read_only and not _is_read_only_sql(query):
            raise RuntimeError(
                "Remote PostgreSQL connections are read-only; refusing non-read query"
            )

    def execute(self, query, data=None):
        self._validate_read_only_query(query)
        return super().execute(query, data)

    def fetch(self, query, data=None):
        self._validate_read_only_query(query)
        return super().fetch(query, data)

    def truncate_app_tables(self):
        self._validate_read_only_query("TRUNCATE TABLE app_tables")
        return super().truncate_app_tables()

    def close(self):
        try:
            super().close()
        finally:
            self.tunnel.close()



def get_db_connection():
    """Get DB instance with credentials from environment variables."""
    db_client = None
    try:
        db_settings = get_db_settings()
        db_client = PostgresDB(
            host=db_settings["host"],
            port=db_settings["port"],
            database=db_settings["database"],
            user=db_settings["user"],
            password=db_settings["password"],
        )
        db_client.apply_local_migrations()
        return db_client
    except Exception as e:
        logger.error(f"[DB] Failed to initialize DB connection: {e}")
        if db_client is not None:
            try:
                db_client.close()
            except Exception:
                pass
        raise


def get_remote_db_connection_via_ssh(
    ssh_host,
    ssh_port,
    ssh_username,
    ssh_password,
    db_settings=None,
):
    """Open a dedicated SSH tunnel and return a Postgres client through it."""
    resolved_db_settings = dict(db_settings or get_db_settings())
    tunnel = SSHTunnelForwarder(
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        ssh_password=ssh_password,
        remote_host=resolved_db_settings["host"],
        remote_port=resolved_db_settings["port"],
    )
    tunnel.start()
    try:
        return SSHTunneledPostgresDB(
            tunnel=tunnel,
            database=resolved_db_settings["database"],
            user=resolved_db_settings["user"],
            password=resolved_db_settings["password"],
            read_only=True,
        )
    except Exception:
        tunnel.close()
        raise


# Usage example:
if __name__ == "__main__":
    db = get_db_connection()
    
    # INSERT
    # db.execute("INSERT INTO users (name, email) VALUES (%s, %s)", ("John", "john@example.com"))
    
    # UPDATE
    # db.execute("UPDATE users SET name = %s WHERE id = %s", ("Jane", 1))
    
    # DELETE
    # db.execute("DELETE FROM users WHERE id = %s", (1,))
    
    # SELECT
    # results = db.fetch("SELECT * FROM users WHERE name LIKE %s", ("%John%",))
    # for row in results:
    #     print(row['name'], row['email'])
    
    db.close()
