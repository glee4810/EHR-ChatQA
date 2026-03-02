import json
import ast
from typing import Dict, Any
from sqlalchemy import text
from sqlalchemy.engine import Engine
from pydantic import BaseModel, Field

class ValueSubstringSearch(BaseModel):
    engine: Engine = Field(..., description="The engine to retrieve sample values from.")

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, table: str, column: str, value: str, k: int = 100) -> str:
        if isinstance(k, str):
            k = int(k)        
        try:
            pattern = f"%{value}%"
            with self.engine.connect() as connection:
                # Step 1: Count total matching distinct values
                count_query = text(
                    f"SELECT COUNT(DISTINCT {column}) FROM {table} WHERE {column} LIKE :pattern COLLATE NOCASE"
                )
                count_result = connection.execute(count_query, {"pattern": pattern})
                n = count_result.scalar()
                if n is None:
                    return f"Error: Unable to count matches in {table}.{column} for '{value}'."
                
                # Step 2: Retrieve up to k matching distinct values
                query = text(
                    f"SELECT DISTINCT {column} FROM {table} WHERE {column} LIKE :pattern COLLATE NOCASE LIMIT {k}"
                )
                res = connection.execute(query, {"pattern": pattern})
                matching_vals = [row[0] for row in res if row[0] is not None]
                matching_vals = list(set(matching_vals))  # Ensure uniqueness

                if not matching_vals:
                    return f"No values in {table}.{column} contain '{value}'."
                
                # Step 3: Construct the response
                base_response = f"Values in {table}.{column} containing '{value}': {matching_vals}."
                return base_response
        except Exception as e:
            return f"Error retrieving matching values: {str(e)}"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "value_substring_search",
                "description": "Retrieve up to k values from a column that contains the specified substring.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "The table name."},
                        "column": {"type": "string", "description": "The column name."},
                        "value": {"type": "string", "description": "The substring to search for."},
                        "k": {"type": "integer", "description": "The maximum number of values to return. Default is 100."},
                    },
                    "required": ["table", "column", "value"],
                },
            },
        }