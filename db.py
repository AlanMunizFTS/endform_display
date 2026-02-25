import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from utilities.log import get_logger
from settings import get_db_settings

logger = get_logger()


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
                allow_repeat=True,
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
            logger.info("[DB] Connection successful", allow_repeat=True)
        except Exception as e:
            logger.error(f"[DB] Connection error: {e}", allow_repeat=True)
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
    
    def close(self):
        """Close all connections"""
        if self.connection_pool:
            self.connection_pool.closeall()

    def insert_img(self, img_name):
        """Insert image name into images table"""
        query = "INSERT INTO img_results (name) VALUES (%s)"
        return self.execute(query, (img_name,))



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
        logger.error(f"[DB] Failed to initialize DB connection: {e}", allow_repeat=True)
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
