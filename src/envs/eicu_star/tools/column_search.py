from typing import Dict, Any
from sqlalchemy import text
from sqlalchemy.engine import Engine
from pydantic import BaseModel, Field

class ColumnSearch(BaseModel):
    engine: Engine = Field(..., description="The engine to retrieve schema and sample rows for.")

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, table_names: str) -> str:
        result = []
        # Split the comma-separated table names and iterate over them
        for table in table_names.split(','):
            table = table.strip()
            try:
                with self.engine.connect() as conn:
                    # Fetch table schema
                    query_schema = f"PRAGMA table_info({table});"
                    schema_rows = conn.execute(text(query_schema)).fetchall()
                    schema = [tuple(row) for row in schema_rows] if schema_rows else []

                    # Fetch foreign keys
                    query_foreign_keys = f"PRAGMA foreign_key_list({table});"
                    fk_rows = conn.execute(text(query_foreign_keys)).fetchall()
                    foreign_keys = [tuple(row) for row in fk_rows] if fk_rows else []

                    # Fetch unique constraints
                    query_unique_keys = f"PRAGMA index_list({table});"
                    unique_rows = conn.execute(text(query_unique_keys)).fetchall()
                    unique_keys_list = [tuple(row) for row in unique_rows] if unique_rows else []
                    # PRAGMA index_list returns: (seq, name, unique, origin, partial)
                    # We're filtering for those with origin 'u'
                    unique_keys = [uk[1] for uk in unique_keys_list if uk[3] == 'u']
                    unique_keys_only = []
                    for key in unique_keys:
                        query = f"PRAGMA index_info('{key}');"
                        index_info_rows = conn.execute(text(query)).fetchall()
                        if index_info_rows:
                            index_info = [tuple(row) for row in index_info_rows]
                            unique_keys_only.append(index_info[0][2])  # third element is the column name

                    # Fetch sample rows (limit to 3)
                    query_sample = f"SELECT * FROM {table} LIMIT 3;"
                    sample_rows = conn.execute(text(query_sample)).fetchall()
                    samples = [tuple(row) for row in sample_rows] if sample_rows else []

                # Build schema string for the table
                schema_str = f"CREATE TABLE {table} ("
                columns_list = []
                for col in schema:
                    # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
                    col_name = col[1]
                    col_type = col[2].upper().replace("INT", "INTEGER") if col[2] else ""
                    if 'TIMESTAMP' in col_type:
                        col_type = 'TIMESTAMP'
                    not_null = "NOT NULL" if col[3] else ""
                    columns_list.append(f"\n\t{col_name} {col_type} {not_null}".rstrip())
                schema_str += ",".join(columns_list)

                # Add primary keys if defined
                primary_keys = [col[1] for col in schema if col[5]]
                if primary_keys:
                    schema_str += f",\n\tPRIMARY KEY ({', '.join(primary_keys)})"

                # Add foreign keys
                for fk in foreign_keys:
                    # PRAGMA foreign_key_list returns: (id, seq, table, from, to, on_update, on_delete, match)
                    schema_str += f",\n\tFOREIGN KEY ({fk[3]}) REFERENCES {fk[2]} ({fk[4]})"

                # Add unique constraints
                for unique_col in unique_keys_only:
                    schema_str += f",\n\tUNIQUE ({unique_col})"

                schema_str = schema_str.rstrip(',\n') + '\n)'

                # Build sample rows string
                column_names = [col[1] for col in schema]
                sample_rows_str = f"\n/*\n3 rows from {table} table:\n" + "\t".join(column_names) + "\n"
                sample_rows_str += "\n".join(["\t".join(map(str, row)) for row in samples]) + "\n*/"

                result.append(schema_str + sample_rows_str)
            except Exception:
                result.append(f"Error: table_names {{'{table}'}} not found in database")
        return "\n\n\n".join(result)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Provides metadata about the tool.
        
        Returns:
            Dict[str, Any]: A dictionary containing the tool's name, description, and parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": "column_search",
                "description": "Get the columns of a specific table and its sample rows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_names": {
                            "type": "string",
                            "description": "A comma-separated list of table names to retrieve schema and sample rows for."
                        }
                    },
                    "required": ["table_names"]
                }
            }
        }