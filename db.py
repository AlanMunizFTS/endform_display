import select
import socketserver
import threading

import paramiko
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from utilities.log import get_logger
from settings import get_db_settings

logger = get_logger()


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
    def __init__(self, host, port, database, user, password):
        """Initialize PostgreSQL connection with connection pool"""
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        try:
            logger.info(
                f"[DB] Connecting to PostgreSQL at {host}:{port}, database={database}, user={user}",
            )
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
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
                    "TRUNCATE TABLE piece_result, classified_images, img_results "
                    "RESTART IDENTITY CASCADE"
                )
                return 3
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


class SSHTunneledPostgresDB(PostgresDB):
    def __init__(self, tunnel, database, user, password):
        self.tunnel = tunnel
        try:
            super().__init__(
                host="127.0.0.1",
                port=tunnel.local_port,
                database=database,
                user=user,
                password=password,
            )
        except Exception:
            self.tunnel.close()
            raise

    def close(self):
        try:
            super().close()
        finally:
            self.tunnel.close()



def get_db_connection():
    """Get DB instance with credentials from environment variables."""
    try:
        db_settings = get_db_settings()
        return PostgresDB(
            host=db_settings["host"],
            port=db_settings["port"],
            database=db_settings["database"],
            user=db_settings["user"],
            password=db_settings["password"],
        )
    except Exception as e:
        logger.error(f"[DB] Failed to initialize DB connection: {e}")
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
