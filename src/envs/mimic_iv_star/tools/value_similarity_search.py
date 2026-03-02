import json
from typing import Dict, Any
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

class ValueSimilaritySearch(BaseModel):
    vector_store: FAISS = Field(..., description="The vector store for similarity search.")

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, table: str, column: str, value: str, k: int = 100) -> str:
        if isinstance(k, str):
            k = int(k)        
        value = value.lower()
        try:
            results = self.vector_store.similarity_search(
                query=value,
                k=k,
                filter={"table": table, "column": column},
                fetch_k=self.vector_store.index.ntotal
            )
            docs_text = [doc.page_content for doc in results]
            if len(results) == 0:
                return f"No matches found in {table}.{column} for '{value}'."
            elif len(docs_text) == 1:
                return (
                    f"I found one close match in {table}.{column}: '{docs_text[0]}'."
                )
            else:
                return (
                    f"I found close matches in {table}.{column}: {docs_text}."
                )
        except Exception as e:
            return f"Error performing similarity search: {str(e)}"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "value_similarity_search",
                "description": "Perform a semantic similarity search (embedding-based) to find up to k values similar to a given value in a specified column. Supported columns: hospitaladmissions: ['admissiontype', 'admitsource', 'dischargedestination'], diagnosiscodes: ['description'], procedurecodes: ['description'], medicationorders: ['medicationname'], clinicalitemtypes: ['itemname'], labtesttypes: ['itemname'], microbiologyresults: ['specimentype', 'testname', 'organismname']",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "The table name."},
                        "column": {"type": "string", "description": "The column name."},
                        "value": {"type": "string", "description": "The value to search for similar matches."},
                        "k": {"type": "integer", "description": "The number of similar values to return. Default is 100."},
                    },
                    "required": ["table", "column", "value"],
                },
            },
        }