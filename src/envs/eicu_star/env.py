import os
import json
from typing import Optional, List
from src.types import Task
from src.envs.base import Env
from src.envs.eicu_star.tools.table_search import TableSearch
from src.envs.eicu_star.tools.column_search import ColumnSearch
from src.envs.eicu_star.tools.sql_execute import SQLExecute
from src.envs.eicu_star.tools.value_substring_search import ValueSubstringSearch
from src.envs.eicu_star.tools.value_similarity_search import ValueSimilaritySearch
from src.envs.eicu_star.tools.web_search import WebSearch
from src.utils import initialize_vector_store, parse_model_name
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

class eICUStarEnv(Env):
    def __init__(
        self,
        user_strategy: str,
        user_model: str,
        user_temperature: float,
        task_type: str,
        task_index: str,
        api_base: Optional[str] = None,
        db_id: str = 'eicu_star',
        db_path: str = "src/envs/eicu_star/eicu_star.sqlite",
        embedding_model: str = "text-embedding-3-large",
        retry_reason: Optional[List[str]] = None
    ):
        self.db_id = db_id
        assert os.path.exists(db_path), f"Database file does not exist: {db_path}"
        self.folder_path = os.path.dirname(__file__)
        if user_strategy == 'human':
            tasks = None
        else:
            with open(os.path.join(self.folder_path, f"eval_{task_type}.jsonl"), "r") as f:
                tasks = [Task(**kwargs) for kwargs in json.load(f)]         

        from ..rules import rules
        if task_type == "incre":
            from ..rules import task_type_incremental
            rules += '\n\n' + task_type_incremental
        elif task_type == "adapt":
            from ..rules import task_type_adaptive
            rules += '\n\n' + task_type_adaptive
        with open(os.path.join(self.folder_path, "db_rules.txt"), "r") as f:
            db_rule = f.read()
        rules += '\n\n' + db_rule

        engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(f"file:{db_path}?immutable=1", uri=True), poolclass=NullPool)
        table_search = TableSearch(engine=engine)
        column_search = ColumnSearch(engine=engine)
        sql_execute = SQLExecute(engine=engine)
        value_substring_search = ValueSubstringSearch(engine=engine)
        faiss_path = 'src/envs/eicu_star/faiss_index_eicu_star-'+parse_model_name(embedding_model)
        columns_to_retrieve = {
            "allergy_reaction": ["drug_name", "allergy_name"],
            "condition": ["condition_name"],
            "fluid_balance": ["fluid_label"],
            "lab": ["lab_name"],
            "prescription": ["drug_name"],
            "icupatient": ["ethnicity", "hospital_admission_source"],
            "treatment": ["treatment_name"]            
        }
        vector_store = initialize_vector_store(engine, embedding_model, faiss_path, columns_to_retrieve)
        value_similarity_search = ValueSimilaritySearch(vector_store=vector_store)
        web_search = WebSearch()

        tools = [
            table_search,
            column_search,
            sql_execute,
            value_substring_search,
            value_similarity_search,
            web_search
        ]

        super().__init__(
            tools=tools,
            tasks=tasks,
            user_strategy=user_strategy,
            user_model=user_model,
            user_temperature=user_temperature,
            db_path=db_path,
            task_type=task_type,
            task_index=task_index,
            rule=rules,
            api_base=api_base,
            retry_reason=retry_reason
        )
