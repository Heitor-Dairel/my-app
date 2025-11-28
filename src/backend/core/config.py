import os
import asyncio
import oracledb
import aiofiles
from dotenv import load_dotenv
from typing import Any, Final, Literal, AsyncGenerator, Callable
from pathlib import Path
from src.backend.utils import HDPrint

load_dotenv()

DB_USER: Final[str | None] = os.getenv("DB_USER")
DB_PASSWORD: Final[str | None] = os.getenv("DB_PASSWORD")
DB_DSN: Final[str | None] = os.getenv("DB_DSN")


class ConnectDB:
    r"""
    Async Oracle database connector and query executor.

    Provides methods to connect, close, execute queries from template files,
    and map results to dictionaries with optional formatting.

    Class Attributes:
        PATH_FILE (Path): Base path to SQL template files.
        MESSAGES (dict[str, str]): Standard success messages for DML operations.
        DICT_STYLE (dict[str, Callable]): Mapping for header formatting styles.
    """

    PATH_FILE: Final[Path] = Path.cwd() / "src" / "backend" / "database" / "template"
    MESSAGES: Final[dict[str, str]] = {
        "INSERT": "inserted successfully",
        "UPDATE": "updated successfully",
        "DELETE": "deleted successfully",
    }
    DICT_STYLE: Final[dict[str, Callable[[str], str]]] = {
        "upper": lambda x: x.upper(),
        "lower": lambda x: x.lower(),
        "title": lambda x: x,
    }

    def __init__(self) -> None:
        """
        Initialize a new ConnectDB instance with default attributes.

        Attributes:
            _user (str | None): Database username from environment variables.
            _password (str | None): Database password from environment variables.
            _dsn (str | None): Database DSN from environment variables.
            _conn (oracledb.AsyncConnectionPool | None): Active async connection, initially None.
            _rows (list[tuple[Any, ...]] | tuple[Any, ...] | None): Query result rows.
            _header (list[tuple[Any, ...]] | tuple[Any, ...] | None): Column metadata.
            _sql (str | None): Last loaded SQL statement.
        """

        self._user: str | None = DB_USER
        self._password: str | None = DB_PASSWORD
        self._dsn: str | None = DB_DSN
        self._conn: oracledb.AsyncConnectionPool | None = None
        self._rows: list[tuple[Any, ...]] | tuple[Any, ...] | None = None
        self._header: list[tuple[Any, ...]] | tuple[Any, ...] | None = None
        self._sql: str | None = None

    @property
    def user(self) -> str | None:
        r"""
        Return the database username.

        Property:
            user (str | None): Provides read-only access to the database username.

        Notes:
            - Returns None if the username has not been set.
        """

        return self._user

    @property
    def password(self) -> str | None:
        r"""
        Return the database password.

        Property:
            password (str | None): Provides read-only access to the database password.

        Notes:
            - Returns None if the password has not been set.
        """

        return self._password

    @property
    def dsn(self) -> str | None:
        r"""
        Return the database DSN (Data Source Name).

        Property:
            dsn (str | None): Provides read-only access to the configured database DSN.

        Notes:
            - Returns None if the DSN has not been set.
        """

        return self._dsn

    @property
    def conn(self) -> oracledb.AsyncConnectionPool | None:
        r"""
        Return the active database connection.

        Property:
            conn (oracledb.AsyncConnectionPool | None): Provides read-only access to the active asynchronous database connection.

        Notes:
            - Returns None if no connection has been established.
        """

        return self._conn

    @property
    def rows(self) -> list[tuple[Any, ...]] | tuple[Any, ...] | None:
        r"""
        Return the rows fetched from the last executed query.

        Property:
            rows (list[tuple[Any, ...]] | tuple[Any, ...] | None): Provides read-only access to the results of the last executed query.

        Notes:
            - Returns None if no query has been executed.
        """

        return self._rows

    @property
    def header(self) -> list[tuple[Any, ...]] | tuple[Any, ...] | None:
        r"""
        Return the column headers of the last SELECT query.

        Property:
            header (list[tuple[Any, ...]] | tuple[Any, ...] | None): Provides read-only access to the column headers.

        Notes:
            - Returns None if no SELECT query has been executed.
        """

        return self._header

    @property
    def sql(self) -> str | None:
        r"""
        Return the last loaded SQL statement.

        Property:
            sql (str | None): Provides read-only access to the last loaded SQL statement.

        Notes:
            - Returns None if no SQL statement has been loaded.
        """

        return self._sql

    async def connect(self) -> None:
        r"""
        Establish an asynchronous connection to the Oracle database.

        Process:
            1. Use stored user, password, and DSN to open a connection.
            2. Assign the connection to `_conn`.

        Raises:
            oracledb.Error: If connection cannot be established.
        """

        self._conn = await oracledb.create_pool_async(
            user=self._user,
            password=self._password,
            dsn=self._dsn,
            min=2,
            max=50,
            increment=1,
        ).acquire()

    async def close(self) -> None:
        r"""
        Close the active database connection.

        Process:
            1. If `_conn` exists, close it asynchronously.
            2. Set `_conn` to None.
        """

        if self._conn:
            await self._conn.close()
        self._conn = None

    @staticmethod
    async def _templatequery(template_file: str, file_path: str) -> str | None:
        r"""
        Load a SQL template from the filesystem asynchronously.

        Process:
            1. Build the full file path using `PATH_FILE`, `template_file`, and `file_path`.
            2. Open the file asynchronously with UTF-8 encoding.
            3. Read the entire SQL content into a string.
            4. Return the SQL string.

        Args:
            template_file (str): Name of the folder containing the SQL template (e.g., "SELECT", "INSERT").
            file_path (str): Name of the SQL file inside the folder.

        Returns:
            (str | None): The content of the SQL file as a string, or None if the file is empty.

        Raises:
            FileNotFoundError: If the specified SQL file does not exist.
            IOError: If the file cannot be read.
        """

        file_name: Path = ConnectDB.PATH_FILE / template_file / file_path

        async with aiofiles.open(file_name, mode="r", encoding="utf8") as sql:
            result: str = await sql.read()
        return result

    @staticmethod
    def _recurrence(
        headers: list[tuple[Any, ...]] | tuple[Any, ...] | None,
        results: list[tuple[Any, ...]] | tuple[Any, ...] | None,
        headers_style: Literal["upper", "lower", "title"] = "title",
    ) -> list[dict[str, Any]] | None:
        r"""
        Convert SQL query results into a list of dictionaries using column headers.

        This method maps each row of query results to a dictionary, where keys are
        derived from column names (`headers`) and values correspond to row data.
        The header names can be formatted according to the specified style.

        Steps:
            1. Format header names based on `headers_style` ("upper", "lower", or "title").
            2. Ensure `results` is a list of tuples (normalize single tuple inputs).
            3. Zip each result row with formatted headers to create dictionaries.
            4. Return a list of these dictionaries, or None if no results are provided.

        Args:
            headers (list[tuple[Any, ...]] | tuple[Any, ...] | None):
                Metadata from a database cursor containing column names.
            results (list[tuple[Any, ...]] | tuple[Any, ...] | None):
                Query results as tuples, either a single row or multiple rows.
            headers_style (Literal["upper", "lower", "title"], optional):
                Case formatting for header names. Defaults to "title".

        Returns:
            (list[dict[str, Any]] | None):
                A list of dictionaries mapping headers to row values, or None if
                `results` is empty.
        """

        if headers:

            headers_names: list[str] = [
                ConnectDB.DICT_STYLE[headers_style](h[0]) for h in headers
            ]

        if results:
            rows_result: list[tuple[Any, ...]] = (
                [results] if isinstance(results, tuple) else results
            )
            result_query: list[dict[str, Any]] = [
                dict(zip(headers_names, i)) for i in rows_result
            ]
            return result_query

        return None

    async def select(
        self,
        template: Literal["SELECT", "UPDATE", "INSERT", "DELETE"],
        query_file: str,
        all_rows: bool = False,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | str | None:
        r"""
        Execute a SQL query from a template file with transaction management.

        Process:
            1. Verify that `self.conn` is a valid database connection.
            2. Load the SQL query from the specified template file.
            3. Execute the SQL query with optional parameters.
            4. If the query is a `SELECT`:
                - Store column headers in `self.header`.
                - Fetch all rows if `all_rows=True`, otherwise fetch a single row.
                - Map results into a list of dictionaries using `_recurrence`.
            5. If the query is `INSERT`, `UPDATE`, or `DELETE`:
                - Commit the transaction.
                - Return a confirmation message from `MESSAGES`.
            6. In case of any exception during execution:
                - Roll back the transaction.
                - Re-raise the exception.

        Args:
            template (Literal["SELECT", "UPDATE", "INSERT", "DELETE"]):
                Type of SQL operation to execute.
            query_file (str):
                Name of the SQL file inside the template directory.
            all_rows (bool, optional):
                Whether to fetch all rows (True) or only the first (False). Defaults to False.
            parameters (dict[str, Any] | None, optional):
                Parameters to bind to the SQL query. Defaults to None.

        Attributes Set:
            self.flag_template (bool): True if the operation is a `SELECT`, otherwise False.
            self.sql (str | None): SQL string loaded from the template file.
            self.header (list[tuple[Any, ...]] | None): Column metadata for `SELECT` queries.
            self.rows (list[tuple[Any, ...]] | tuple[Any, ...] | None): Query results.

        Returns:
            (list[dict[str, Any]] | str | None):
                - List of dictionaries mapping column names to values (for `SELECT`).
                - Success message string (for `INSERT`, `UPDATE`, `DELETE`).
                - None if no rows are returned.

        Raises:
            ConnectionError: If no active database connection exists.
            Exception: If query execution fails (transaction is rolled back before re-raising).
        """

        if self._conn:
            self._sql = await ConnectDB._templatequery(template, query_file)
            async with self._conn.cursor() as cur:
                try:
                    await cur.execute(self._sql, parameters)
                    if template == "SELECT":
                        self._header = cur.description
                        if all_rows:
                            self._rows = await cur.fetchall()
                        else:
                            self._rows = await cur.fetchone()
                        result: list[dict[str, Any]] | None = ConnectDB._recurrence(
                            self._header, self._rows
                        )
                        return result

                    await self._conn.commit()
                    return ConnectDB.MESSAGES[template]
                except Exception:
                    await self._conn.rollback()
                    raise


async def session_db() -> AsyncGenerator[ConnectDB, None]:
    r"""
    Async context generator for database sessions.

    Process:
        1. Create a new `ConnectDB` instance.
        2. Open a database connection.
        3. Yield the active database session for use.
        4. Ensure the connection is closed when finished.

    Yields:
        ConnectDB: An active database session.
    """

    db = ConnectDB()

    await db.connect()

    try:
        yield db

    finally:
        await db.close()


if __name__ == "__main__":

    async def main():
        db = ConnectDB()

        await db.connect()

        result = await db.select("SELECT", "teste.sql", False, {"teste": 1})

        if result is not None:
            HDPrint(*result).print_json()
        await db.close()

    asyncio.run(main())


# python -W ignore -m src.backend.core.config
