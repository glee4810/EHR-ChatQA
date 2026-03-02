import os
import json
from typing import Dict, Any, List
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

class WebSearch(BaseModel):
    search_tool: TavilySearchResults = Field(
        default_factory=lambda: TavilySearchResults(max_results=10),
        description="The Tavily search tool instance."
    )
    
    class Config:
        arbitrary_types_allowed = True

    def invoke(self, query: str) -> str:
        response = self.search_tool.invoke(query)
        return json.dumps([doc['content'] for doc in response], ensure_ascii=False, indent=2)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a web search to obtain external knowledge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                    },
                    "required": ["query"],
                },
            },
        }
    