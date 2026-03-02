from typing import Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine
from sqlalchemy import inspect

class TableSearch(BaseModel):
    engine: Engine = Field(..., description="The engine to list tables from.")

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, tool_input: str = "") -> str:
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        return ", ".join(tables)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "table_search",
                "description": "Get the list of table names in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_input": {
                            "type": "string",
                            "description": "An empty string; no input required.",
                        }
                    },
                    "required": []
                }
            }
        }
