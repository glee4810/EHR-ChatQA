from typing import Dict, Any
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from func_timeout import func_timeout, FunctionTimedOut

class SQLExecute(BaseModel):
    engine: Engine = Field(..., description="The engine to execute queries on.")

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, query: str, k: int = 100, timeout: int = 60) -> str:
        if isinstance(k, str):
            k = int(k)
        if isinstance(timeout, str):
            timeout = int(timeout)            
        result = ""
        try:
            def execute_query():
                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    return result.fetchall()
            
            result = func_timeout(timeout, execute_query)
            n = len(result)
            base_response = str(result[:k])
            if n > k:
                additional = n - k
                base_response += (
                    f"\n\nNote: There are {additional} results not shown (out of {n} total results)."
                )
        except FunctionTimedOut:
            base_response = f"Error: Query execution timed out after {timeout} seconds"
        except SQLAlchemyError as e:
            """Format the error message"""
            base_response = f"Error: {e}"
        return base_response

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "sql_execute",
                "description": "Execute a SQL query against the database and get back the result. If the query is not correct, an error message will be returned. The maximum number of results to return is 100.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A valid SQL query to execute."
                        },
                        "k": {
                            "type": "integer",
                            "description": "The maximum number of results to return. Default is 100."
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Maximum execution time in seconds before the query is terminated. Default is 60 seconds."
                        }
                    },
                    "required": ["query"]
                }
            }
        }