import os
import shutil
import sqlite3
from typing import TypedDict, List, Dict, Any, Optional, AsyncIterator
from langgraph.graph import StateGraph, END
from openai import OpenAI
import json
import pandas as pd
import re

# -------------------------------
# OpenAI Client
# -------------------------------
client = OpenAI(api_key="")

DEFAULT_MODEL_NAME = "gpt-4o-mini" 

# -------------------------------
# DB CONFIG
# -------------------------------
DB_PATH = "ecommerce.db"


# -------------------------------
# Schema
# -------------------------------

SCHEMA_INFO = """
Database Schema for E-commerce System:

1. customers
   - customer_id (TEXT): Unique customer identifier
   - customer_unique_id (TEXT): Unique customer identifier across datasets
   - customer_zip_code_prefix (INTEGER): Customer zip code
   - customer_city (TEXT): Customer city
   - customer_state (TEXT): Customer state

2. orders
   - order_id (TEXT): Unique order identifier
   - customer_id (TEXT): Foreign key to customers
   - order_status (TEXT): Order status (delivered, shipped, etc.)
   - order_purchase_timestamp (TEXT): When the order was placed
   - order_approved_at (TEXT): When payment was approved
   - order_delivered_carrier_date (TEXT): When order was handed to carrier
   - order_delivered_customer_date (TEXT): When customer received the order
   - order_estimated_delivery_date (TEXT): Estimated delivery date

3. order_items
   - order_id (TEXT): Foreign key to orders
   - order_item_id (INTEGER): Item sequence number within order
   - product_id (TEXT): Foreign key to products
   - seller_id (TEXT): Foreign key to sellers
   - shipping_limit_date (TEXT): Shipping deadline
   - price (REAL): Item price
   - freight_value (REAL): Shipping cost

4. order_payments
   - order_id (TEXT): Foreign key to orders
   - payment_sequential (INTEGER): Payment sequence number
   - payment_type (TEXT): Payment method (credit_card, boleto, etc.)
   - payment_installments (INTEGER): Number of installments
   - payment_value (REAL): Payment amount

5. order_reviews
   - review_id (TEXT): Unique review identifier
   - order_id (TEXT): Foreign key to orders
   - review_score (INTEGER): Review score (1-5)
   - review_comment_title (TEXT): Review title
   - review_comment_message (TEXT): Review message
   - review_creation_date (TEXT): When review was created
   - review_answer_timestamp (TEXT): When review was answered

6. products
   - product_id (TEXT): Unique product identifier
   - product_category_name (TEXT): Product category (in Portuguese)
   - product_name_lenght (REAL): Product name length
   - product_description_lenght (REAL): Product description length
   - product_photos_qty (REAL): Number of product photos
   - product_weight_g (REAL): Product weight in grams
   - product_length_cm (REAL): Product length in cm
   - product_height_cm (REAL): Product height in cm
   - product_width_cm (REAL): Product width in cm

7. sellers
   - seller_id (TEXT): Unique seller identifier
   - seller_zip_code_prefix (INTEGER): Seller zip code
   - seller_city (TEXT): Seller city
   - seller_state (TEXT): Seller state

8. geolocation
   - geolocation_zip_code_prefix (INTEGER): Zip code prefix
   - geolocation_lat (REAL): Latitude
   - geolocation_lng (REAL): Longitude
   - geolocation_city (TEXT): City name
   - geolocation_state (TEXT): State code

9. product_category_name_translation
   - product_category_name (TEXT): Category name in Portuguese
   - product_category_name_english (TEXT): Category name in English
"""

# -------------------------------
# Agent State
# -------------------------------
class AgentState(TypedDict, total=False):
    # 원 질문 (파이프라인 전체에서 참조용)
    question: str              # 사용자가 처음 보낸 자연어 질문
    final_answer: str          # (원하면) 최종 요약 답변을 넣을 수 있는 필드

    # 대화 컨텍스트 (LangGraph 노드들이 사용하는 기본 구조)
    messages: List[Dict[str, Any]]
    user_query: str

    # 스키마 정보
    schema_info: str

    # --- Guardrail / Router ---
    guardrail_in_scope: bool
    guardrail_reason: str
    guardrail_raw: str

    intent: str                # READ_ONLY / SCHEMA_CHANGE / EXPERIMENT_START / OUT_OF_SCOPE
    router_reason: str
    router_raw: str

    # --- World / Branch 관리 ---
    current_world_id: str
    worlds: Dict[str, Dict[str, Any]]  # world_id -> meta

    branch_plan: Dict[str, Any]        # experiment_planner_agent 결과
    branch_sql: Dict[str, List[str]]   # world_id -> [sql1, sql2, ...]
    branch_results: Dict[str, Dict[str, Any]]   # world_id -> {metrics, samples, ...}
    branch_sql_progress: Dict[str, int]         # world_id -> 다음 실행할 SQL index
    failed_worlds: List[str]                    # 실행 실패/포기된 world 목록

    # --- 에러 / 재시도 ---
    last_error: Optional[str]
    error_world_id: Optional[str]
    error_sql: Optional[str]
    error_sql_index: Optional[int]
    needs_error_handling: bool
    error_retry_counts: Dict[str, int]
    error_agent_raw: str

    # --- 평가 / 커밋 ---
    evaluation_message: str
    evaluation_raw: str
    chosen_world_id: Optional[str]
    commit_result_message: str

    # --- READ_ONLY 전용 ---
    read_only_sql: str
    read_only_result_message: str

    # --- OUT_OF_SCOPE 전용 ---
    final_message: str

class BranchManager:
    """
    BranchManager for SQLite (DB-per-world 방식).

    - base_db_path: 메인라인 DB 파일 (예: "ecommerce.db")
    - world_dir: 브랜치용 DB 파일을 저장할 디렉토리 (예: "worlds")

    world 구조 예시:
    self.worlds = {
        "main": {
            "status": "mainline",
            "parent": None,
            "description": "Mainline database",
            "db_path": "ecommerce.db",
        },
        "world_1": {
            "status": "active",
            "parent": "main",
            "description": "5% coupon strategy",
            "db_path": "worlds/world_1.db",
        },
        ...
    }
    """

    def __init__(self, base_db_path: str, world_dir: str = "worlds") -> None:
        self.base_db_path = base_db_path
        self.world_dir = world_dir
        os.makedirs(self.world_dir, exist_ok=True)

        if not os.path.exists(self.base_db_path):
            raise FileNotFoundError(
                f"Base DB file not found: {self.base_db_path}. "
                "Make sure ecommerce.db is created first."
            )

        # world 메타데이터
        self.worlds: Dict[str, Dict[str, Any]] = {
            "main": {
                "status": "mainline",
                "parent": None,
                "description": "Mainline database",
                "db_path": os.path.abspath(self.base_db_path),
            }
        }

        # world ID 생성용 카운터
        self._world_counter: int = 0

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------
    def _next_world_id(self) -> str:
        self._world_counter += 1
        return f"world_{self._world_counter}"

    def _get_db_path(self, world_id: str) -> str:
        info = self.worlds.get(world_id)
        if not info:
            raise ValueError(f"Unknown world_id: {world_id}")
        return info["db_path"]

    def _connect(self, world_id: str) -> sqlite3.Connection:
        db_path = self._get_db_path(world_id)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------
    def get_worlds(self) -> Dict[str, Dict[str, Any]]:
        """현재 world 메타데이터 사본을 반환."""
        return {k: dict(v) for k, v in self.worlds.items()}

    def init_state_for_agent(self) -> Dict[str, Any]:
        """
        LangGraph AgentState 초기화용 헬퍼.

        예:
            state.update(branch_manager.init_state_for_agent())
        """
        return {
            "current_world_id": "main",
            "worlds": self.get_worlds(),
        }

    def create_world(self, parent_id: str = "main", description: str = "") -> str:
        """
        parent_id를 기반으로 새로운 world를 만든다.
        - parent world의 DB 파일을 그대로 복사해서 새 world DB를 생성.
        - 반환값: 새로운 world_id (예: "world_1")
        """
        if parent_id not in self.worlds:
            raise ValueError(f"Parent world '{parent_id}' does not exist")

        world_id = self._next_world_id()
        parent_db_path = self._get_db_path(parent_id)

        world_db_path = os.path.abspath(
            os.path.join(self.world_dir, f"{world_id}.db")
        )
        shutil.copy2(parent_db_path, world_db_path)

        self.worlds[world_id] = {
            "status": "active",
            "parent": parent_id,
            "description": description or f"Branch from {parent_id}",
            "db_path": world_db_path,
        }

        return world_id

    def run_sql(self, world_id: str, sql: str) -> Dict[str, Any]:
        """
        주어진 world에서 SQL을 실행하고 결과를 반환.

        - sql: 하나 또는 여러 개의 SQL 문을 포함할 수 있음.
          (간단한 구현: 전체를 하나의 문으로 보고 실행)
        - SELECT면 rows/columns를 포함하여 반환
        - DDL/DML이면 영향받은 row 수 등만 반환

        반환 형식 예:
        {
          "world_id": "world_1",
          "statement": "SELECT ...",
          "type": "select",
          "rows": [ {...}, {...} ],
          "columns": ["col1", "col2"],
          "rowcount": 2
        }
        """
        sql_stripped = sql.strip()
        if not sql_stripped:
            raise ValueError("Empty SQL string")

        conn = self._connect(world_id)
        cur = conn.cursor()

        first_token = sql_stripped.lstrip().split(None, 1)[0].upper()
        is_select = first_token in ("SELECT", "WITH")  # 🔥 핵심 수정

        result: Dict[str, Any] = {
            "world_id": world_id,
            "statement": sql_stripped,
            "type": "select" if is_select else "other",
        }

        try:
            cur.execute(sql_stripped)
            if is_select:
                rows = cur.fetchall()
                columns = [col[0] for col in cur.description] if cur.description else []
                result["columns"] = columns
                result["rows"] = [dict(r) for r in rows]
                result["rowcount"] = len(result["rows"])
            else:
                conn.commit()
                result["rowcount"] = cur.rowcount
        finally:
            cur.close()
            conn.close()

        return result

    def commit_world(self, world_id: str) -> None:
        """
        world_id의 DB 상태를 메인라인(ecommerce.db)에 커밋.

        - world_id의 DB 파일 내용을 base_db_path로 덮어쓴다.
        - world 메타데이터에서 world_id는 status="committed"로 표시.
        - main world는 여전히 'main'이라는 ID를 유지하되,
          그 DB 파일 내용이 갱신된 것으로 본다.
        """
        if world_id == "main":
            # main은 이미 베이스이므로 별도 커밋 불필요
            return

        if world_id not in self.worlds:
            raise ValueError(f"World '{world_id}' does not exist")

        src_db_path = self._get_db_path(world_id)
        dst_db_path = os.path.abspath(self.base_db_path)

        shutil.copy2(src_db_path, dst_db_path)

        # 메타데이터 갱신
        self.worlds[world_id]["status"] = "committed"
        # main world의 db_path는 그대로 base_db_path를 가리키므로 자동으로 최신
        self.worlds["main"]["db_path"] = dst_db_path

    def rollback_world(self, world_id: str) -> None:
        """
        world_id를 롤백(폐기) 처리.
        - main은 롤백할 수 없음.
        - world DB 파일 삭제 + status="rolled_back"으로 변경
        """
        if world_id == "main":
            raise ValueError("Cannot rollback the main world")

        info = self.worlds.get(world_id)
        if not info:
            raise ValueError(f"World '{world_id}' does not exist")

        db_path = info.get("db_path")
        if db_path and os.path.exists(db_path):
            os.remove(db_path)

        info["status"] = "rolled_back"
        # 원하면 여기서 self.worlds.pop(world_id)로 완전히 제거할 수도 있음

    def get_schema(self, world_id: str) -> str:
        """
        해당 world의 실제 SQLite 스키마를 문자열로 반환.
        - 새로운 테이블/칼럼이 생겼을 수도 있으므로,
          SCHEMA_INFO(정적 문자열) 대신 이걸 써도 된다.
        """
        conn = self._connect(world_id)
        cur = conn.cursor()

        # 모든 테이블 리스트
        cur.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row["name"] for row in cur.fetchall()]

        schema_lines: List[str] = []
        for tbl in tables:
            schema_lines.append(f"Table: {tbl}")
            cur.execute(f"PRAGMA table_info({tbl})")
            cols = cur.fetchall()
            for c in cols:
                # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
                schema_lines.append(
                    f"  - {c['name']} ({c['type']})"
                )
            schema_lines.append("")  # 빈 줄

        cur.close()
        conn.close()

        return "\n".join(schema_lines) or "(no tables found)"
branch_manager = BranchManager(DB_PATH, world_dir="worlds")
def build_initial_state(user_message: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "messages": [{"role": "user", "content": user_message}],
        "user_query": user_message,
        "schema_info": SCHEMA_INFO,
        # 그 외 필요 필드 초기화
        "branch_sql": {},
        "branch_results": {},
        "error_retry_counts": {},
    }
    # world 관련 필드 채우기
    state.update(branch_manager.init_state_for_agent())
    return state

AGENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "guardrail": {
        "model": DEFAULT_MODEL_NAME,
        "description": "Check if the user query is in-scope for the ecommerce SQL assistant.",
        "system_prompt": f"""
You are the Guardrail Agent for an E-commerce Text-to-SQL assistant.

Your job:
1. Read the user's question.
2. Decide if it can be answered using ONLY the following SQLite database: "ecommerce.db".
3. The database represents an online marketplace with customers, orders, products, sellers, geolocation, and product reviews.

IN-SCOPE:
- Questions about orders, payments, customers, products, sellers, reviews, and shipping.
- Aggregations like total revenue, average review score, top-selling categories, etc.
- What-if or experimental questions that can be simulated with SQL on this schema, including:
  - Trying different discount or coupon strategies.
  - Comparing multiple pricing policies.
  - Exploring different customer segments.
- Requests to:
  - CREATE, ALTER, or DROP tables/columns (schema changes).
  - INSERT, UPDATE, or DELETE rows.
  - Create speculative branches, run experiments, and then COMMIT the best branch back to the main database,
    as long as everything is based ONLY on this database content and SQL operations.

OUT-OF-SCOPE:
- Questions requiring external or real-time data (e.g., live stock prices, external APIs, other websites).
- Questions unrelated to this database (e.g., general chit-chat, personal advice).
- Programming questions not related to querying or transforming this database.
- Tasks that require external ML models, optimization solvers, or business knowledge beyond what can
  reasonably be approximated using SQL queries on this database.

IMPORTANT:
- DO NOT mark a question as out-of-scope just because it mentions "strategy", "coupon", "branch", "experiment",
  or "commit". As long as the entire task can be simulated using SQL queries and updates on this database,
  it is IN-SCOPE.

Database schema:
{SCHEMA_INFO}

Reply ONLY in JSON:
{{
  "in_scope": true or false,
  "reason": "<short explanation>"
}}
""".strip(),
    },

    "router": {
        "model": DEFAULT_MODEL_NAME,
        "description": "Classify the user query into routing intent.",
        "system_prompt": f"""
You are the Router Agent for an E-commerce Text-to-SQL system.

You receive:
- The user's question about the ecommerce database.
- (Optionally) some conversation history.

Your job:
- Classify the query into EXACTLY ONE of the following intent labels:
  1. READ_ONLY        -> Pure data analysis / SELECT queries only.
  2. SCHEMA_CHANGE    -> Requests to create/alter/drop tables or columns, or other structural changes.
  3. EXPERIMENT_START -> Requests to explore multiple hypothetical strategies or "what-if" scenarios,
                         which should be run in separate speculative branches (worlds).

Use the database schema below to understand what is possible:

{SCHEMA_INFO}

Return a short JSON-looking snippet in plain text like:
intent: READ_ONLY
reason: <one-line explanation>

Do NOT write SQL in this step.
""".strip(),
    },
    "analysis_agent": {
        "role": "Data Analyst",
        "system_prompt": "You are a helpful data analyst that explains database query results in natural language with clear insights.",
    },
    "viz_agent": {
        "role": "Visualization Specialist", 
        "system_prompt": "You are a data visualization expert. Generate clean, executable Plotly code without any markdown formatting or explanations.",
    },

    "experiment_planner": {
        "model": DEFAULT_MODEL_NAME,
        "description": "Design 3 speculative branches (strategies) for an ecommerce experiment.",
        "system_prompt": (
            # f-접두사 없는 일반 문자열
            "You are the Experiment Planner Agent for an E-commerce Text-to-SQL system.\n\n"
            "Goal:\n"
            "- The router has decided that the user's request is an EXPERIMENT_START.\n"
            "- You must design **exactly 3 speculative branches (strategies)** that explore different\n"
            "  'what-if' hypotheses on the following ecommerce database:\n\n"
            # 여기서만 SCHEMA_INFO를 f-string으로 끼워넣기
            f"{SCHEMA_INFO}\n\n"
            "The database represents an online marketplace with:\n"
            "- customers, orders, order_items, order_payments, order_reviews,\n"
            "- products, sellers, geolocation,\n"
            "- product_category_name_translation.\n\n"
            "Examples of experiments:\n"
            "- Different discount / coupon strategies.\n"
            "- Different shipping policies or thresholds.\n"
            "- Different ways of selecting target customer segments.\n"
            "- Different pricing or bundling strategies.\n\n"
            "Your tasks:\n"
            "1. Read the user's question carefully.\n"
            "2. Propose EXACTLY 3 distinct branches (strategies) that are all plausible responses\n"
            "   to the user's request, but differ in their assumptions or actions.\n"
            "3. For each branch:\n"
            "   - Define a short 'branch_id' (e.g., \"b1\", \"b2\", \"b3\").\n"
            "   - Give a human-readable 'name' (e.g., \"5% coupon on electronics\").\n"
            "   - Describe its 'hypothesis' (what this branch assumes or tests).\n"
            "   - Provide an ordered list 'operations' of NATURAL-LANGUAGE steps\n"
            "     that will later be converted into SQL, e.g.:\n"
            "       - \"Apply a 5% discount to all orders in December 2017\"\n"
            "       - \"Recalculate total revenue under this discount policy\"\n"
            "       - \"Compute average review_score for affected orders\"\n\n"
            "4. Decide a GLOBAL primary evaluation metric:\n"
            "   - Example: \"total_revenue\", \"avg_order_value\", \"return_rate\",\n"
            "              \"num_orders\", \"num_active_customers\", etc.\n"
            "5. Optionally provide a list of secondary metrics.\n\n"
            "IMPORTANT:\n"
            "- Do NOT write SQL here.\n"
            "- Describe 'operations' in natural language only.\n"
            "- All branches must be feasible using the given schema (tables/columns above).\n\n"
            "Output format:\n"
            "Reply ONLY in JSON with this structure (no extra text):\n\n"
            "{\n"
            "  \"branches\": [\n"
            "    {\n"
            "      \"branch_id\": \"b1\",\n"
            "      \"name\": \"<short strategy name>\",\n"
            "      \"hypothesis\": \"<what this branch tests>\",\n"
            "      \"operations\": [\n"
            "        \"<step 1 in natural language>\",\n"
            "        \"<step 2 in natural language>\"\n"
            "      ]\n"
            "    },\n"
            "    {\n"
            "      \"branch_id\": \"b2\",\n"
            "      \"name\": \"...\",\n"
            "      \"hypothesis\": \"...\",\n"
            "      \"operations\": [ \"...\" ]\n"
            "    },\n"
            "    {\n"
            "      \"branch_id\": \"b3\",\n"
            "      \"name\": \"...\",\n"
            "      \"hypothesis\": \"...\",\n"
            "      \"operations\": [ \"...\" ]\n"
            "    }\n"
            "  ],\n"
            "  \"primary_metric\": \"<one metric name>\",\n"
            "  \"secondary_metrics\": [\"<metric1>\", \"<metric2>\"]\n"
            "}\n"
        ).strip(),
    },
    "sql_agent": {
        "model": DEFAULT_MODEL_NAME,
        "description": "Generate SQLite SQL for the given operation(s) and branch.",
        "system_prompt": f"""
    You are the SQL Generation Agent for an E-commerce Text-to-SQL system.

    Goal:
    - Given:
    - The user's question,
    - The selected intent (READ_ONLY / SCHEMA_CHANGE / EXPERIMENT_START),
    - And (for experiments) a specific branch's natural-language 'operations' description,
    - You must generate one or more **SQLite-compatible SQL statements**
    that operate on the following database schema:

    {SCHEMA_INFO}

    IMPORTANT CONSTRAINTS (READ CAREFULLY):
    - You MUST use ONLY the columns that actually exist in the schema above.
    - There is NO column that stores precomputed order totals.
    When you need total revenue, you MUST compute it from `order_items`
    (e.g., `SUM(order_items.price + order_items.freight_value)`),
    possibly joined with `orders` to apply date filters.
    - Do NOT use columns like `order_total`, `total_amount`, `grand_total`, etc.,
    because they do NOT exist in the schema.
    - When you need revenue, compute it using:
    `SUM(order_items.price + order_items.freight_value)`
    joined with `orders` for date filters.

    - Each SQL statement is executed independently.
    If you define a CTE with WITH (e.g., `WITH top_customers AS (...)`), you MUST:
    - Put ALL CTE definitions and the final SELECT in a SINGLE statement.
    - Example (GOOD):
        WITH a AS (...),
            b AS (...)
        SELECT ...
    - NEVER split one logical CTE query into multiple pieces.
        For example, this is FORBIDDEN:
        "WITH a AS (...),"
        "b AS (...)" 
        "SELECT ... FROM b"
        Each element in any SQL list (e.g., `"sql": ["..."]`) MUST be a COMPLETE, standalone SQL statement.

    - Do NOT use window functions or advanced SQL features.
    - FORBIDDEN: `NTILE`, `ROW_NUMBER`, `RANK`, `DENSE_RANK`, or any `... OVER (...)` clauses.
    - Use only basic SQL that is supported by SQLite: `SELECT`, `JOIN`, `WHERE`,
        `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`, and simple CTEs with `WITH`.

    - In `HAVING` clauses, you MUST NOT use SELECT aliases.
    - FORBIDDEN (BAD):
        SELECT SUM(x) AS total_value
        ...
        HAVING total_value >= 100
    - Instead, REPEAT the aggregate expression:
        HAVING SUM(x) >= 100

    - For EXPERIMENT_START when you are asked to return multiple SQL statements (e.g., in a JSON `"sql"` array):
    - Each string in the array MUST be a full, valid SQL statement that can be executed on its own.
    - NEVER break a single SQL statement across multiple entries.
    - If you need multiple steps (e.g., create a temp table, then select from it), each step must be one full statement.
    
    - For EXPERIMENT_START, You must include at least one CREATE TABLE or INSERT statement. And you also MUST end with a final SELECT statement that computes the primary metric.
    - You MUST NOT put discount or scaling arithmetic directly inside SELECT expressions.
  - Forbidden examples (DO NOT DO THIS):
      SELECT SUM(cart_value * 0.95) AS total_revenue ...
      SELECT price * 0.9 AS discounted_price ...
  - Instead, you MUST first materialize the discounted value using an UPDATE (or CREATE TABLE ... AS SELECT ...)
    and then SELECT the resulting column without any arithmetic.
    For example (GOOD pattern):
      UPDATE orders
      SET discounted_value = cart_value * 0.95
      WHERE ...;

      SELECT SUM(discounted_value) AS total_revenue, COUNT(*) AS order_count
      FROM orders
      WHERE ...;

    General rules:
    - Use only tables and columns that exist in the schema.
    - Prefer explicit JOINs using primary/foreign key relationships:
    - customers.customer_id = orders.customer_id
    - orders.order_id = order_items.order_id
    - orders.order_id = order_payments.order_id
    - orders.order_id = order_reviews.order_id
    - order_items.product_id = products.product_id
    - order_items.seller_id = sellers.seller_id
    - products.product_category_name = product_category_name_translation.product_category_name
    - When filtering by time, use the appropriate *_timestamp or *_date columns in 'orders' or 'order_reviews'.
    - When grouping by categories or states, use:
    - customer_state from customers
    - seller_state from sellers
    - product_category_name or product_category_name_english from the translation table.

    Safety rules:
    - For READ_ONLY: generate ONLY SELECT statements.
    - For EXPERIMENT_START or SCHEMA_CHANGE:
    - You may generate CREATE TABLE / INSERT / UPDATE / DELETE / ALTER TABLE statements,
        but these will always be executed in an isolated branch (world), never on the mainline directly.
    - For experiment branches, it is often helpful to materialize intermediate results in branch-local tables
        using CREATE TABLE ... AS SELECT ... or INSERT INTO ... SELECT ... before doing a final SELECT
        to compute aggregate metrics.

    Output format:
    - Return ONLY the SQL statements as plain text.
    - If multiple statements are required, separate them with a semicolon and a newline.
    - Do NOT add natural language explanation in this step.
    """.strip(),
    },

    "error_agent": {
        "model": DEFAULT_MODEL_NAME,
        "description": "Fix SQL when execution fails, given the error message.",
        "system_prompt": f"""
You are the SQL Error-Correction Agent for an E-commerce Text-to-SQL system.

You will receive:
- The original SQL query that failed.
- The error message returned by SQLite.
- The database schema (see below).

Your job:
- Identify why the SQL failed (e.g., wrong column name, invalid table alias, syntax error).
- Return a **corrected SQL query** that should succeed on this schema:

{SCHEMA_INFO}

Rules:
- Keep the user's original intent unchanged.
- Fix only what is necessary (table/column names, joins, aliases, GROUP BY issues, etc.).
- Do NOT invent new tables or columns that are not present in the schema.
- Always return ONLY the corrected SQL (no explanation), so it can be executed directly.

If the SQL is fundamentally impossible with this schema, return the string:
-- IMPOSSIBLE

In that case, the system will stop retrying and report failure to the user.
""".strip(),
    },

    "evaluate_agent": {
        "model": DEFAULT_MODEL_NAME,
        "description": "Compare multiple branch/world results and recommend the best strategy.",
        "system_prompt": f"""
You are the Evaluation Agent for speculative branches in an E-commerce Text-to-SQL system.

Context:
- The system may create multiple speculative branches (worlds), each representing a different strategy.
  Examples:
    - Different coupon/discount strategies,
    - Different pricing rules,
    - Different segmentation of customers or products.
- For each branch, you will be given:
  - A 'strategy description' (what was changed),
  - One or more numeric metrics (e.g., total_revenue, avg_order_value, return_rate),
  - Possibly sample rows from the simulation.

Your tasks:
1. Compare all branches based on the provided metrics.
2. Explain in clear language how each strategy performed
   (e.g., which one maximizes revenue, which one increases returns).
3. Recommend ONE branch as the best candidate to commit back to the mainline.
4. Highlight important trade-offs (for example, higher revenue but much higher return_rate).

The underlying database is:

{SCHEMA_INFO}

Output format:
- Brief bullet-point comparison for each strategy.
- A final line: "recommended_world_id: <world_id>" that the system can parse.
- Keep it concise but clear enough for a non-technical business user.

Do NOT generate SQL here.
""".strip(),
    },
}

def _extract_user_query_from_state(state: "AgentState") -> str:
    """
    state["user_query"]가 없으면 messages 리스트에서 마지막 user 메시지를 찾아 사용.
    없으면 빈 문자열 반환.
    """
    if "user_query" in state and state["user_query"]:
        return state["user_query"]

    messages = state.get("messages", []) or []
    # messages는 [{"role": "user"/"assistant", "content": "..."}] 형태라고 가정
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))

    return ""


def guardrails_agent(state: "AgentState") -> "AgentState":
    """
    Guardrail Agent:
    - 유저 질문이 ecommerce.db 스키마로 답할 수 있는지(in-scope) 판단.
    - 결과를 state["guardrail_in_scope"], state["guardrail_reason"] 에 저장.

    이후 Router 노드에서 이 값을 보고 out-of-scope면
    바로 종료 응답을 하거나 fallback 로직을 태우면 됨.
    """
    user_query = _extract_user_query_from_state(state)

    # 아무 질문도 없으면 그냥 in-scope로 두고 패스
    if not user_query.strip():
        state["guardrail_in_scope"] = True
        state["guardrail_reason"] = "Empty query; treating as in-scope by default."
        return state

    cfg = AGENT_CONFIG["guardrail"]
    system_prompt = cfg["system_prompt"]

    # JSON으로 반드시 응답하도록 추가 지시
    system_prompt_with_json = (
        system_prompt
        + """

IMPORTANT:
Reply ONLY in JSON with the following structure (no extra text):

{
  "in_scope": true or false,
  "reason": "<short explanation>"
}
"""
    )

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt_with_json},
            {"role": "user", "content": user_query},
        ],
        temperature=0.0,
    )

    raw_content = resp.choices[0].message.content.strip()

    # 디버깅을 위해 원문도 state에 남겨두자 (선택)
    state["guardrail_raw"] = raw_content

    in_scope = True
    reason = ""

    try:
        data: Dict[str, Any] = json.loads(raw_content)
        if isinstance(data.get("in_scope"), bool):
            in_scope = data["in_scope"]
        reason = str(data.get("reason", "")).strip()
    except Exception:
        # 실패하면 일단 in_scope=True로 두고, reason은 LLM 원문
        in_scope = True
        reason = f"Failed to parse JSON guardrail response. Raw: {raw_content}"

    state["guardrail_in_scope"] = in_scope
    state["guardrail_reason"] = reason

    return state

def router_agent(state: "AgentState") -> "AgentState":
    """
    Router Agent:
    - 유저 질문을 보고 intent를 분류한다.
      - READ_ONLY
      - SCHEMA_CHANGE
      - EXPERIMENT_START
    - guardrail이 out-of-scope라고 판단한 경우 intent를 OUT_OF_SCOPE로 설정.

    결과:
    - state["intent"] = 위 네 가지 중 하나 (또는 OUT_OF_SCOPE)
    - state["router_reason"] = 라우팅 이유 (LLM 설명)
    """
    # 1) guardrail 결과가 이미 out-of-scope면 바로 종료용 intent 설정
    if state.get("guardrail_in_scope") is False:
        state["intent"] = "OUT_OF_SCOPE"
        state["router_reason"] = state.get(
            "guardrail_reason",
            "Marked as out-of-scope by guardrail agent.",
        )
        return state

    user_query = _extract_user_query_from_state(state)

    # 질문이 없으면 그냥 READ_ONLY로 기본값
    if not user_query.strip():
        state["intent"] = "READ_ONLY"
        state["router_reason"] = "Empty query; defaulting to READ_ONLY."
        return state

    cfg = AGENT_CONFIG["router"]
    system_prompt = cfg["system_prompt"]

    system_prompt_with_json = (
        system_prompt
        + """

IMPORTANT:
Reply ONLY in JSON with the following structure (no extra text):

{
  "intent": "READ_ONLY" | "SCHEMA_CHANGE" | "EXPERIMENT_START" ,
  "reason": "<short explanation>"
}
"""
    )

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt_with_json},
            {"role": "user", "content": user_query},
        ],
        temperature=0.0,
    )

    raw_content = resp.choices[0].message.content.strip()
    state["router_raw"] = raw_content  # 디버깅용

    intent = "READ_ONLY"
    reason = ""

    try:
        data: Dict[str, Any] = json.loads(raw_content)
        raw_intent = str(data.get("intent", "")).strip().upper()
        reason = str(data.get("reason", "")).strip()

        # 허용된 intent만 통과
        allowed_intents = {
            "READ_ONLY",
            "SCHEMA_CHANGE",
            "EXPERIMENT_START",
        }
        if raw_intent in allowed_intents:
            intent = raw_intent
        else:
            # 이상한 값이면 READ_ONLY로 폴백
            intent = "READ_ONLY"
            if not reason:
                reason = f"Invalid intent '{raw_intent}', defaulting to READ_ONLY."
    except Exception:
        # JSON 파싱 실패 시 READ_ONLY로 폴백
        intent = "READ_ONLY"
        reason = f"Failed to parse router JSON response. Raw: {raw_content}"

    state["intent"] = intent
    state["router_reason"] = reason

    return state

def experiment_planner_agent(state: "AgentState") -> "AgentState":
    """
    Experiment Planner Agent

    - Router에서 intent == 'EXPERIMENT_START' 일 때 호출.
    - 사용자의 질문을 바탕으로, EXACTLY 3개의 브랜치(전략)를 설계한다.
    - 각 브랜치는 natural-language 'operations' 목록을 포함하고,
      나중에 SQL agent가 이 operations를 보고 SQL을 생성하게 된다.

    결과:
    - state["branch_plan"] = {
          "branches": [...],
          "primary_metric": "...",
          "secondary_metrics": [...]
      }
    """

    # EXPERIMENT_START가 아니면 그냥 아무것도 안 하고 통과
    if state.get("intent") != "EXPERIMENT_START":
        return state

    user_query = _extract_user_query_from_state(state)

    cfg = AGENT_CONFIG["experiment_planner"]
    system_prompt = cfg["system_prompt"]

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.2,
    )

    raw_content = resp.choices[0].message.content.strip()
    state["experiment_planner_raw"] = raw_content  # 디버깅용

    try:
        data: Dict[str, Any] = json.loads(raw_content)

        # 최소한 branches는 리스트여야 함
        branches = data.get("branches", [])
        if not isinstance(branches, list) or len(branches) == 0:
            raise ValueError("No valid 'branches' field in experiment planner output.")

        # primary_metric, secondary_metrics 기본값 처리
        primary_metric = data.get("primary_metric") or "total_revenue"
        secondary_metrics = data.get("secondary_metrics")
        if not isinstance(secondary_metrics, list):
            secondary_metrics = []

        state["branch_plan"] = {
            "branches": branches,
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
        }

    except Exception as e:
        # 파싱 실패 시: 아주 단순한 fallback plan 생성
        # (그래프가 터지지 않도록 최소 구조를 제공)
        state["branch_plan"] = {
            "branches": [
                {
                    "branch_id": "b1",
                    "name": "Baseline scenario",
                    "hypothesis": "Baseline strategy as close as possible to current behavior.",
                    "operations": [
                        "Analyze current orders, payments, and revenues without any changes."
                    ],
                },
                {
                    "branch_id": "b2",
                    "name": "Aggressive discount",
                    "hypothesis": "Discount-focused strategy to boost order volume.",
                    "operations": [
                        "Simulate a 10 percent discount on all orders in the last 3 months.",
                        "Recalculate total revenue and order counts under this discount."
                    ],
                },
                {
                    "branch_id": "b3",
                    "name": "Targeted discount",
                    "hypothesis": "Target discounts only to customers with high past spending.",
                    "operations": [
                        "Identify top 20 percent customers by historical spending.",
                        "Simulate a 10 percent discount only for these customers.",
                        "Recalculate revenue and order counts under this policy."
                    ],
                },
            ],
            "primary_metric": "total_revenue",
            "secondary_metrics": ["num_orders"],
            "planner_fallback_error": str(e),
        }

    return state

def branch_world_creator_agent(state: "AgentState") -> "AgentState":
    """
    Experiment Planner가 만든 branch_plan을 기반으로
    각 브랜치에 대응하는 world(DB 복제본)를 만드는 에이전트.

    역할:
    - state["branch_plan"]["branches"] 안에 있는 각 branch에 대해:
        - branch_id 기준으로 새 world_id를 하나씩 생성
        - BranchManager.create_world(...)를 호출해 실제 DB 파일 복제
        - branch_plan["branch_to_world"][branch_id] = world_id 매핑 추가
    - state["worlds"] 를 BranchManager의 world 메타데이터로 갱신

    주의:
    - 이미 branch_to_world에 등록된 branch_id는 다시 만들지 않아서
      재실행해도 idempotent하게 동작함.
    """

    # EXPERIMENT_START / SCHEMA_CHANGE 가 아닐 때는 그냥 통과
    if state.get("intent") not in ("EXPERIMENT_START", "SCHEMA_CHANGE"):
        return state

    plan: Dict[str, Any] = state.get("branch_plan") or {}
    branches: List[Dict[str, Any]] = plan.get("branches", [])
    if not branches:
        # 설계된 브랜치가 없다면 할 일이 없음
        return state

    # 기존 매핑이 있으면 이어서 쓰고, 없으면 새 dict
    branch_to_world: Dict[str, str] = plan.get("branch_to_world", {}) or {}
    # AgentState 안의 worlds 메타데이터 (없으면 BranchManager 기준으로 초기화)
    worlds_meta: Dict[str, Dict[str, Any]] = state.get("worlds") or branch_manager.get_worlds()

    # parent world는 기본적으로 main을 기준으로 브랜치를 딴다고 가정
    parent_world_id = state.get("current_world_id", "main")
    if parent_world_id not in branch_manager.get_worlds():
        parent_world_id = "main"

    for br in branches:
        branch_id = br.get("branch_id")
        name = br.get("name", "")
        if not branch_id:
            continue

        # 이미 world가 만들어져 있으면 스킵 (idempotent)
        if branch_id in branch_to_world:
            continue

        # 브랜치 설명을 world description으로 활용
        desc = f"{branch_id}: {name}".strip()

        # 실제 world (DB 복제본) 생성
        world_id = branch_manager.create_world(parent_id=parent_world_id, description=desc)

        # 매핑 및 메타데이터 갱신
        branch_to_world[branch_id] = world_id
        # BranchManager의 최신 worlds 정보 가져와서 state에 반영
        worlds_meta[world_id] = branch_manager.get_worlds()[world_id]

    # state에 반영
    plan["branch_to_world"] = branch_to_world
    state["branch_plan"] = plan
    state["worlds"] = worlds_meta

    return state

def sql_agent_experiment(state: "AgentState") -> "AgentState":
    """
    SQL Agent (Experiment mode)

    역할:
    - intent == 'EXPERIMENT_START' 일 때,
      experiment_planner_agent가 만든 branch_plan을 읽어서
      각 branch의 'operations'를 기반으로 SQL 문 리스트를 생성한다.
    - 결과는 state["branch_sql"][world_id] = [sql1, sql2, ...]로 저장.

    기대 입력 (AgentState):
    - state["intent"] == "EXPERIMENT_START"
    - state["branch_plan"] = {
          "branches": [
              {
                  "branch_id": "b1",
                  "name": "...",
                  "hypothesis": "...",
                  "operations": ["...", "..."],
              },
              ...
          ],
          "primary_metric": "...",
          "secondary_metrics": [...],
          "branch_to_world": {"b1": "world_1", ...}  # 선택적
      }

    출력:
    - state["branch_sql"] (Dict[str, List[str]]) 에 world_id 기준 SQL 리스트를 채워 넣는다.
    """
    # EXPERIMENT_START가 아니면 아무것도 안 하고 패스
    if state.get("intent") != "EXPERIMENT_START":
        return state

    plan: Dict[str, Any] = state.get("branch_plan") or {}
    branches: List[Dict[str, Any]] = plan.get("branches", [])
    branch_to_world: Dict[str, str] = plan.get("branch_to_world", {}) or {}

    if not branches:
        # 설계된 브랜치가 없으면 할 일이 없음
        return state

    # 이미 존재하는 branch_sql 있으면 이어쓰기
    branch_sql: Dict[str, List[str]] = state.get("branch_sql") or {}
    state["branch_sql"] = branch_sql  # 참조 유지

    cfg = AGENT_CONFIG["sql_agent"]
    base_system_prompt = cfg["system_prompt"]
    model_name = cfg["model"]

    user_query = _extract_user_query_from_state(state)

    for branch in branches:
        branch_id = branch.get("branch_id")
        name = branch.get("name", "")
        hypothesis = branch.get("hypothesis", "")
        operations = branch.get("operations", [])

        if not branch_id or not operations:
            continue

        # world_id 매핑: 없으면 branch_id 자체를 world 키로 사용 (fallback)
        world_id = branch_to_world.get(branch_id, branch_id)

        # 이미 해당 world에 SQL이 있다면 (재실행 방지용) 스킵
        if world_id in branch_sql and branch_sql[world_id]:
            continue

        # operations를 자연스럽게 나열
        operations_text = "\n".join(f"- {op}" for op in operations)

        # 이번 호출만을 위한 system prompt override:
        # 원래 sql_agent system_prompt는 "plain SQL만" 요구하지만,
        # 여기서는 JSON으로 받아서 파싱할 거라 형식을 덮어쓴다.
        system_prompt = (
            base_system_prompt
            + """

OVERRIDE OUTPUT FORMAT FOR THIS CALL:

You are generating SQL for a SPECULATIVE EXPERIMENT BRANCH.
You must consider the user's question AND the specific branch description and operations.

For THIS call only, DO NOT return plain SQL text.
Instead, reply ONLY in JSON with the following structure (no extra text):

{
  "sql": [
    "<first SQL statement>",
    "<second SQL statement>",
    "... (if needed)"
  ]
}
"""
        )

        # user 메시지 구성
        user_content = f"""
User question:
{user_query}

You are generating SQL for the following experiment branch:

- branch_id: {branch_id}
- branch_name: {name}
- hypothesis: {hypothesis}

Operations to implement in this branch (in order):
{operations_text}

Please generate a sequence of SQLite-compatible SQL statements that, when executed
in a clean branch database cloned from the main ecommerce.db, will implement
these operations and then compute the metrics needed to evaluate this strategy.
""".strip()

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )

        raw_content = resp.choices[0].message.content.strip()

        # 디버깅용: 어떤 응답이 나왔는지 저장
        sql_agent_debug_key = f"sql_agent_raw_{world_id}"
        state[sql_agent_debug_key] = raw_content

        sql_list: List[str] = []

        # 1) JSON 파싱 시도
        try:
            data = json.loads(raw_content)
            if isinstance(data, dict):
                maybe_sql = data.get("sql", [])
                if isinstance(maybe_sql, list):
                    sql_list = [str(s).strip() for s in maybe_sql if str(s).strip()]
        except Exception:
            sql_list = []

        # 2) JSON 파싱 실패한 경우, 혹시 그냥 SQL 텍스트를 줬다면 세미콜론으로 fallback split
        if not sql_list:
            # 세미콜론 기준으로 자르고, 마지막 빈 조각 제거
            parts = [p.strip() for p in raw_content.split(";")]
            sql_list = [p + ";" for p in parts if p]

        # 최종 SQL 리스트를 state에 저장
        if sql_list:
            print("================")
            print("World ID:", world_id)
            print("SQL: ", sql_list)
            print("================")
            branch_sql[world_id] = sql_list

    # state["branch_sql"]는 이미 참조를 유지하고 있음
    return state

def execute_sql_agent(state: "AgentState") -> "AgentState":
    """
    Execute SQL Agent (sequential, per-branch, 재시도/실패 브랜치 인식 버전).

    역할:
    - state["branch_sql"]에 담긴 world별 SQL 리스트를 순차 실행.
    - world별로 진행 상태(몇 번째 SQL까지 성공했는지)를 state["branch_sql_progress"]에 저장.
    - 이미 failed_worlds에 들어간 world는 건너뜀.
    - 실행 중 예외 발생 시:
        - state["last_error"], ["error_world_id"], ["error_sql"], ["error_sql_index"] 설정
        - state["needs_error_handling"] = True
        - 즉시 반환 (Error Agent가 수정하도록)

    모든 브랜치의 남은 SQL을 다 성공적으로 실행하면:
        - state["needs_error_handling"] = False
        - state["branch_results"]에 world별 실행 로그/샘플이 모임
        - 이후 evaluate_agent로 넘어가면 됨.
    """
    # EXPERIMENT_START가 아닐 때는 그냥 통과
    if state.get("intent") != "EXPERIMENT_START":
        return state

    branch_sql: Dict[str, List[str]] = state.get("branch_sql") or {}
    branch_results: Dict[str, Dict[str, Any]] = state.get("branch_results") or {}
    state["branch_results"] = branch_results

    progress: Dict[str, int] = state.get("branch_sql_progress") or {}
    failed_worlds: List[str] = state.get("failed_worlds") or []

    # 에러 플래그 초기화
    state["needs_error_handling"] = False
    state.pop("last_error", None)
    state.pop("error_world_id", None)
    state.pop("error_sql", None)
    state.pop("error_sql_index", None)

    for world_id, sql_list in branch_sql.items():
        if not sql_list:
            continue

        # 이미 실패 처리된 브랜치는 건너뜀
        if world_id in failed_worlds:
            continue

        # 이 world에서 다음으로 실행해야 할 SQL index
        start_idx = progress.get(world_id, 0)
        if start_idx >= len(sql_list):
            # 이미 이 world의 모든 SQL을 돌았음
            continue

        # world 결과 구조 준비
        world_res = branch_results.get(world_id) or {
            "sql_log": [],
            "samples": [],
            "metrics": {},
        }

        for idx in range(start_idx, len(sql_list)):
            sql = (sql_list[idx] or "").strip()
            if not sql:
                # 빈 SQL이면 그냥 건너뜀
                progress[world_id] = idx + 1
                continue

            try:
                res = branch_manager.run_sql(world_id, sql)

                # 전체 로그에 추가
                world_res["sql_log"].append(res)

                # SELECT라면 샘플 저장
                if res.get("type") == "select":
                    rows = res.get("rows") or []
                    if rows:
                        world_res["samples"].append({
                            "statement_index": idx,
                            "statement": res.get("statement", ""),
                            "columns": res.get("columns", []),
                            "rows": rows[:5],
                        })

                # 성공했으니 다음 index로 진행
                progress[world_id] = idx + 1

            except Exception as e:
                # 에러 발생 → Error Agent로 넘기기 위한 정보 저장
                state["last_error"] = str(e)
                state["error_world_id"] = world_id
                state["error_sql"] = sql
                state["error_sql_index"] = idx
                state["needs_error_handling"] = True

                # 지금까지의 world 결과/진행상태 반영
                if not world_res.get("metrics"):
                    world_res["metrics"] = _extract_metrics_from_world_res(world_res)

                branch_results[world_id] = world_res
                state["branch_results"] = branch_results
                state["branch_sql_progress"] = progress
                state["failed_worlds"] = failed_worlds
                return state

        # 이 world는 남은 SQL들을 모두 성공적으로 실행
        if not world_res.get("metrics"):
            world_res["metrics"] = _extract_metrics_from_world_res(world_res)
        branch_results[world_id] = world_res

    # 여기까지 왔다는 것은, 남은 SQL이 있는 모든 world가 성공하거나,
    # 전부 이미 완료/실패 상태라는 뜻
    state["branch_results"] = branch_results
    state["branch_sql_progress"] = progress
    state["failed_worlds"] = failed_worlds
    state["needs_error_handling"] = False
    return state

def error_agent(state: "AgentState") -> "AgentState":
    """
    SQL Error Agent

    역할:
    - execute_sql_agent에서 에러가 발생했을 때 호출.
    - 해당 world의 실패한 SQL을 LLM을 이용해 수정.
    - world 당 최대 3회 재시도:
        - 3회 초과 or LLM이 '-- IMPOSSIBLE' 반환 → 해당 world는 실패 처리.
        - 실패된 world는 failed_worlds에 추가되고, 이후 실행/평가에서 제외된다.

    입력(필요 필드):
    - state["last_error"]: str           (SQLite 에러 메시지)
    - state["error_world_id"]: str
    - state["error_sql_index"]: int
    - state["error_sql"]: str
    - state["branch_sql"]: Dict[str, List[str]]
    - state["error_retry_counts"]: Dict[str, int]  (없으면 자동 생성)

    출력:
    - 수정된 state:
        - branch_sql에 수정된 SQL 반영 또는 failed_worlds에 world 추가.
        - needs_error_handling = False 로 (다시 execute_sql_agent로 돌아가도록).
    """
    # 에러 플래그가 없으면 할 일이 없음
    if not state.get("needs_error_handling"):
        return state

    world_id = state.get("error_world_id")
    error_sql_index = state.get("error_sql_index")
    error_sql = state.get("error_sql")
    error_msg = state.get("last_error")

    if world_id is None or error_sql_index is None or error_sql is None:
        # 정보가 부족하면 그냥 아무 것도 하지 않고 종료
        state["needs_error_handling"] = False
        return state

    branch_sql: Dict[str, List[str]] = state.get("branch_sql") or {}
    sql_list = branch_sql.get(world_id)

    if not sql_list or not (0 <= error_sql_index < len(sql_list)):
        # 수정할 대상이 없으면 이 world를 실패로 처리
        _mark_world_failed(
            state,
            world_id,
            error_msg or "Unknown SQL error (no sql_list)"
        )
        state["needs_error_handling"] = False
        return state

    # 재시도 카운트 갱신
    retry_counts: Dict[str, int] = state.get("error_retry_counts") or {}
    current_retry = retry_counts.get(world_id, 0) + 1
    retry_counts[world_id] = current_retry
    state["error_retry_counts"] = retry_counts

    # 3번 초과하면 이 브랜치는 포기
    if current_retry > 5:
        _mark_world_failed(
            state,
            world_id,
            f"Exceeded max retries for world {world_id}. Last error: {error_msg}",
        )
        state["needs_error_handling"] = False
        return state

    # 여기서부터는 LLM에 수정 요청
    cfg = AGENT_CONFIG["error_agent"]
    system_prompt = cfg["system_prompt"]

    # world별 실제 스키마를 함께 주면 수정 정확도가 올라감
    try:
        world_schema = branch_manager.get_schema(world_id)
    except Exception:
        world_schema = "(failed to fetch world schema)"

    system_prompt_with_schema = (
        system_prompt
        + "\n\nHere is the current schema for THIS branch:\n"
        + world_schema
        + "\n"
    )

    # 원래 실패한 SQL도 코드펜스가 껴 있을 수 있으니, 그대로 보여주되 LLM이 잘 이해하게만 한다.
    user_content = f"""
The following SQL failed when executed in SQLite.

Original SQL:
{error_sql}

SQLite error message:
{error_msg}

Please return a corrected SQL query that keeps the user's intent,
and is valid for the given schema.
Return ONLY the corrected SQL, without any explanation.
""".strip()

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt_with_schema},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    raw_reply = (resp.choices[0].message.content or "").strip()
    # 디버깅용 raw 응답 저장
    state["error_agent_raw"] = raw_reply

    # 코드펜스(```sql ... ``` )가 감싸져 있을 수 있으니 벗겨낸다.
    corrected_sql = _strip_code_fences(raw_reply)

    # LLM이 '-- IMPOSSIBLE' 이라고 하면 이 world는 포기
    if corrected_sql.upper().startswith("-- IMPOSSIBLE") or not corrected_sql:
        _mark_world_failed(
            state,
            world_id,
            f"Error agent marked as impossible. Last error: {error_msg}",
        )
        state["needs_error_handling"] = False
        return state

    # 해당 인덱스에 수정된 SQL 반영
    sql_list[error_sql_index] = corrected_sql
    branch_sql[world_id] = sql_list
    state["branch_sql"] = branch_sql

    # 에러 플래그 클리어 → 다시 execute_sql_agent로 돌려보냄
    state["needs_error_handling"] = False
    # last_error 관련 정보는 남겨도 되고 지워도 됨 (여기선 남겨둔다)
    return state


def evaluate_agent(state: "AgentState") -> "AgentState":
    """
    Evaluation Agent

    역할:
    - branch_plan, branch_results, failed_worlds를 종합해서
      각 브랜치(세계)의 전략/메트릭을 비교하고 설명하는 자연어 메시지를 만든다.
    - 실패한 브랜치는 "이 전략은 실행에 실패했습니다"라고 분리해서 설명.
    - 성공한 브랜치들 중에서 하나의 world_id를 추천하고,
      state["chosen_world_id"]에 저장.
    - 유저에게 "어느 전략을 메인에 커밋할까요?"라고 묻는 질문까지 포함.

    입력(주요 필드):
    - state["branch_plan"] = {
          "branches": [
             {"branch_id": "...", "name": "...", "hypothesis": "...", ...},
             ...
          ],
          "primary_metric": "...",
          "secondary_metrics": [...],
          "branch_to_world": {"b1": "world_1", ...} (선택적)
      }
    - state["branch_results"] = {
          "world_1": {
              "metrics": {...},
              "samples": [...],
              "status": "failed" (optional),
              "failure_reason": "..." (optional),
              ...
          },
          ...
      }
    - state["failed_worlds"] = ["world_2", ...]

    출력:
    - state["chosen_world_id"] = 추천된 world_id (또는 None)
    - state["evaluation_message"] = 유저에게 보여줄 자연어 설명 + 질문
    """

    # EXPERIMENT_START가 아닐 때는 그냥 통과
    if state.get("intent") != "EXPERIMENT_START":
        return state

    plan: Dict[str, Any] = state.get("branch_plan") or {}
    branches: List[Dict[str, Any]] = plan.get("branches", [])
    branch_to_world: Dict[str, str] = plan.get("branch_to_world", {}) or {}
    primary_metric: str = plan.get("primary_metric") or "total_revenue"
    secondary_metrics: List[str] = plan.get("secondary_metrics") or []

    branch_results: Dict[str, Dict[str, Any]] = state.get("branch_results") or {}
    failed_worlds: List[str] = state.get("failed_worlds") or []

    # 브랜치 정보가 없으면 그냥 종료 메시지
    if not branches:
        msg = "실험용 브랜치 정보가 없습니다. 먼저 EXPERIMENT_START 플로우를 실행해 주세요."
        state["evaluation_message"] = msg
        state["chosen_world_id"] = None
        return state

    # LLM에게 넘길 평가 입력 구조 정리
    eval_items: List[Dict[str, Any]] = []

    for idx, branch in enumerate(branches, start=1):
        branch_id = branch.get("branch_id")
        if not branch_id:
            continue

        world_id = branch_to_world.get(branch_id, branch_id)  # fallback
        res = branch_results.get(world_id, {})

        # 상태/실패 여부
        status = res.get("status")
        if not status:
            status = "failed" if world_id in failed_worlds else "ok"

        metrics = res.get("metrics", {})

        eval_items.append(
            {
                "strategy_index": idx,
                "world_id": world_id,
                "branch_id": branch_id,
                "name": branch.get("name", ""),
                "hypothesis": branch.get("hypothesis", ""),
                "status": status,
                "metrics": metrics,
            }
        )

    cfg = AGENT_CONFIG["evaluate_agent"]
    base_system_prompt = cfg["system_prompt"]

    # 출력 형식을 더 엄격하게 지정 (JSON 파싱 + 자연어 설명 둘 다)
    system_prompt = (
        base_system_prompt
        + """

IMPORTANT OUTPUT INSTRUCTIONS (OVERRIDE):

1) 먼저, 사용자에게 보여줄 한국어 설명을 작성하세요.
   - 각 전략을 "전략 1", "전략 2"처럼 번호로 불러 주세요.
   - 각 전략에 대해:
     - world_id, 간단한 전략 이름(name), hypothesis를 요약하고
     - metrics에 들어 있는 주요 지표(예: total_revenue, order_count, return_rate 등)를
       사람이 이해하기 쉬운 문장으로 설명하세요.
   - status가 "failed"인 브랜치는
     - "이 전략은 실행에 실패했습니다." 라고 명시하고,
       후보 전략에서 제외해야 함을 설명하세요.
   - 마지막에는 예시와 비슷한 형식으로 요약해 주세요. 예:
       - 전략 1: 매출 +3%, 반품 +1%
       - 전략 2: 매출 +7%, 반품 +5%
       - 전략 3: 이 전략은 실행에 실패했습니다.

2) 그 다음 줄에, 파싱 가능한 형태로 추천 결과를 적어주세요.
   - 마지막 줄 한 줄에만 다음 형식을 사용합니다:

     recommended_world_id: <world_id 또는 NONE>

   예:
     recommended_world_id: world_2

   모든 성공한 전략이 없거나 추천할 수 없다면:
     recommended_world_id: NONE
"""
    )

    # LLM에 넘길 JSON payload
    user_payload = {
        "primary_metric": primary_metric,
        "secondary_metrics": secondary_metrics,
        "branches": eval_items,
    }

    user_message = (
        "다음은 실험 브랜치(world)와 그 메트릭 정보입니다 (JSON 형식):\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
        + "\n\n위 정보를 바탕으로 각 전략을 비교하고, 최적의 world_id를 하나 추천해 주세요."
    )

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    text = (resp.choices[0].message.content or "").strip()
    state["evaluation_raw"] = text  # 디버깅용

    # 마지막 줄에서 recommended_world_id 파싱
    chosen_world_id = None
    match = re.search(r"recommended_world_id:\s*(\S+)", text)
    if match:
        val = match.group(1).strip()
        if val.upper() != "NONE":
            chosen_world_id = val

    # 혹시 LLM이 실패 브랜치를 추천했으면 방어적으로 무효화
    if chosen_world_id and chosen_world_id in (state.get("failed_worlds") or []):
        chosen_world_id = None

    state["chosen_world_id"] = chosen_world_id
    state["evaluation_message"] = text

    return state

def auto_commit_best_world_agent(state: "AgentState") -> "AgentState":
    """
    자동 commit 모드용 노드.

    - evaluate_agent가 선택한 state["chosen_world_id"]를 보고
      해당 world를 메인 DB에 커밋하고, 나머지 world는 롤백한다.
    - main은 항상 그대로 유지하되, DB 파일 내용만 갱신된다고 가정.
    - 결과 요약을 state["commit_result_message"]에 저장.
    """
    chosen_world_id = state.get("chosen_world_id")
    failed_worlds: List[str] = state.get("failed_worlds") or []

    if not chosen_world_id:
        state["commit_result_message"] = (
            "추천된 브랜치가 없어서 아무 것도 커밋하지 않았습니다."
        )
        return state

    if chosen_world_id in failed_worlds:
        state["commit_result_message"] = (
            f"추천된 브랜치 {chosen_world_id} 가 실패 상태여서 커밋할 수 없습니다."
        )
        return state

    # BranchManager의 world 목록 가져오기
    bm_worlds: Dict[str, Dict[str, Any]] = branch_manager.get_worlds()
    committed = None
    rolled_back: List[str] = []

    for world_id in bm_worlds.keys():
        if world_id == "main":
            continue

        if world_id == chosen_world_id:
            branch_manager.commit_world(world_id)
            committed = world_id
        else:
            branch_manager.rollback_world(world_id)
            rolled_back.append(world_id)

    # AgentState의 worlds도 최신 상태로 맞춰주기
    state["worlds"] = branch_manager.get_worlds()

    if committed is None:
        state["commit_result_message"] = (
            "커밋할 브랜치를 찾지 못했습니다. (선택된 world_id가 잘못되었을 수 있습니다.)"
        )
    else:
        state["commit_result_message"] = (
            f"브랜치 {committed} 를 메인에 커밋하고, 나머지 브랜치 {rolled_back} 는 롤백했습니다."
        )

    return state

def sql_agent_read_only(state: "AgentState") -> "AgentState":
    """
    READ_ONLY intent 전용 SQL 생성 에이전트.

    역할:
    - 자연어 질문을 받아서 SQLite용 SELECT 문 1개를 생성한다.
    - 생성된 SQL은 state["read_only_sql"]에 저장한다.
    - 실제 실행은 execute_sql_read_only_agent 같은 별도 노드에서 처리.

    Router에서:
      intent == "READ_ONLY" 일 때만 호출되도록 연결하는 게 자연스럽다.
    """
    if state.get("intent") != "READ_ONLY":
        return state

    user_query = _extract_user_query_from_state(state)
    if not user_query.strip():
        state["read_only_sql"] = ""
        return state

    cfg = AGENT_CONFIG["sql_agent"]
    base_system_prompt = cfg["system_prompt"]

    # 이 호출에서만 적용되는 제약사항을 system_prompt에 덧붙인다.
    system_prompt = (
        base_system_prompt
        + """

FOR THIS CALL (READ_ONLY MODE):

- You MUST generate exactly ONE SQLite-compatible SELECT statement.
- Do NOT use DDL (CREATE/ALTER/DROP) or DML (INSERT/UPDATE/DELETE).
- The query should answer the user's question as directly as possible.
- Output ONLY the SQL statement as plain text (no explanation).
"""
    )

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.0,
    )

    sql_text = (resp.choices[0].message.content or "").strip()
    state["read_only_sql"] = sql_text

    return state
def sql_agent_schema(state: "AgentState") -> "AgentState":
    """
    SCHEMA_CHANGE intent 전용 SQL 생성 에이전트.

    역할:
    - 사용자의 스키마 변경 요청(테이블/컬럼 생성, 수정 등)을 이해하고,
      해당 변경을 수행할 수 있는 DDL 중심 SQL 리스트를 생성한다.
    - 생성된 SQL 리스트는 state["branch_sql"][world_id] 에 저장된다.
      (여기서 world_id는 보통 현재 world, 기본적으로 "main")

    주의:
    - 이 에이전트는 "어떤 스키마 변경을 할지"만 결정한다.
    - 실제 실행은 execute_sql_agent가 담당한다.
    """
    if state.get("intent") != "SCHEMA_CHANGE":
        return state

    user_query = _extract_user_query_from_state(state)
    if not user_query.strip():
        return state

    cfg = AGENT_CONFIG["sql_agent"]
    base_system_prompt = cfg["system_prompt"]

    # 이 호출에서는 DDL 중심으로, JSON 형식으로 sql 리스트를 돌려달라고 강하게 요구
    system_prompt = (
        base_system_prompt
        + """

FOR THIS CALL (SCHEMA_CHANGE MODE):

- The user is asking for schema-level changes:
  e.g., create new tables, add/drop/rename columns, add indexes, etc.
- You MUST design one or more SQLite-compatible DDL statements (and optional helper DML)
  to implement the requested schema changes on the ecommerce database.

- Examples of allowed statements:
  - CREATE TABLE ...
  - ALTER TABLE ... ADD COLUMN ...
  - ALTER TABLE ... RENAME COLUMN ...
  - CREATE INDEX ...
  - INSERT INTO ... SELECT ...   (if needed to backfill data)
  - UPDATE ...                   (if needed to migrate data)

- You MUST return your answer ONLY in JSON (no extra text), with the format:

{
  "sql": [
    "<first SQL statement>",
    "<second SQL statement>",
    "... (if needed)"
  ]
}
"""
    )

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.2,
    )

    raw = (resp.choices[0].message.content or "").strip()
    # 디버깅용으로 원문 저장해도 좋다
    state["sql_agent_schema_raw"] = raw

    sql_list: List[str] = []

    # 1) JSON 파싱 시도
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            maybe_sql = data.get("sql", [])
            if isinstance(maybe_sql, list):
                sql_list = [str(s).strip() for s in maybe_sql if str(s).strip()]
    except Exception:
        sql_list = []

    # 2) JSON 파싱 실패 시, 혹시 그냥 세미콜론으로 구분된 SQL을 줬다면 fallback
    if not sql_list:
        parts = [p.strip() for p in raw.split(";")]
        sql_list = [p + ";" for p in parts if p]

    if not sql_list:
        # 아무 것도 못 만들었으면 그냥 state만 반환
        return state

    # branch_sql 구조에 현재 world 기준으로 SQL 리스트를 채운다.
    branch_sql: Dict[str, List[str]] = state.get("branch_sql") or {}
    world_id = state.get("current_world_id", "main")

    branch_sql[world_id] = sql_list
    state["branch_sql"] = branch_sql

    return state
def execute_sql_read_only_agent(state: "AgentState") -> "AgentState":
    """
    READ_ONLY intent 전용 SQL 실행 에이전트.

    역할:
    - sql_agent_read_only 가 생성한 state["read_only_sql"] 을
      현재 world (보통 main) 에서 실행한다.
    - SELECT 결과 일부(최대 10행)를 텍스트로 포맷팅해서
      state["read_only_result_message"] 에 저장한다.
    - 동시에 analysis_agent에서 사용할 수 있도록
      state["sql"], state["query_result"] 도 채운다.
    - 에러가 나면 에러 메시지를 그대로 넣어준다.
    """

    # READ_ONLY가 아니면 아무 것도 안 하고 통과
    if state.get("intent") != "READ_ONLY":
        return state

    sql = (state.get("read_only_sql") or "").strip()
    if not sql:
        state["read_only_result_message"] = "생성된 SQL이 없습니다."
        # analysis_agent에서 그대로 써먹을 수 있게 비워둔 값도 넣어둠
        state["sql"] = ""
        state["query_result"] = ""
        return state

    # 어떤 world에서 실행할지: 기본은 main
    world_id = state.get("current_world_id", "main")

    try:
        res = branch_manager.run_sql(world_id, sql)
    except Exception as e:
        msg = (
            "SQL 실행 중 오류가 발생했습니다.\n\n"
            f"World: {world_id}\n"
            f"SQL:\n{sql}\n\n"
            f"에러: {e}"
        )
        state["read_only_result_message"] = msg
        state["sql"] = sql
        state["query_result"] = msg
        return state

    state["sql"] = sql  # analysis / decide_graph_need 에서 사용

    if res.get("type") == "select":
        cols = res.get("columns") or []
        rows = res.get("rows") or []
        head_rows = rows[:10]

        lines = [
            "다음은 생성된 SQL과 결과 일부입니다:",
            "",
            "SQL:",
            sql,
            "",
            "결과 (최대 10행):",
        ]
        result_lines = []

        if not head_rows:
            lines.append("(결과가 없습니다.)")
            result_lines.append("(결과가 없습니다.)")
        else:
            if cols:
                header = " | ".join(cols)
                sep = "-" * len(header)
                lines.append(header)
                lines.append(sep)
                result_lines.append(header)
                result_lines.append(sep)
                for r in head_rows:
                    row_str = " | ".join(str(r.get(c, "")) for c in cols)
                    lines.append(row_str)
                    result_lines.append(row_str)
            else:
                for r in head_rows:
                    row_str = str(r)
                    lines.append(row_str)
                    result_lines.append(row_str)

        state["read_only_result_message"] = "\n".join(lines)
        state["query_result"] = "\n".join(result_lines)

    else:
        msg = (
            "SELECT가 아닌 SQL이 실행되었습니다.\n\n"
            f"World: {world_id}\n"
            f"SQL:\n{sql}\n\n"
            f"영향 받은 행 수: {res.get('rowcount')}"
        )
        state["read_only_result_message"] = msg
        state["query_result"] = msg

    return state


def analysis_agent(state: AgentState) -> AgentState:
    """Generate natural language answer from query results"""
    question = state.get("question", "")
    sql_query = state.get("sql", "")
    query_result = state.get("query_result", "")
    
    prompt = f"""You are a helpful assistant that explains database query results in natural language.

Original Question: {question}

SQL Query Used: {sql_query}

Query Results:
{query_result}

Please provide a clear, concise answer to the original question based on the query results.
Format the answer in a user-friendly way. If the results contain numbers, present them clearly.
If there are multiple queries/results (for multi-part questions), address each part of the question separately.
Use bullet points or numbered lists for multiple answers.

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": AGENT_CONFIG["analysis_agent"]["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    final_answer = response.choices[0].message.content.strip()
    state["final_answer"] = final_answer
    
    return state
def decide_graph_need(state: AgentState) -> AgentState:
    """Decide if a graph visualization would be helpful for the query"""
    question = state.get("question", "")
    query_result = state.get("query_result", "")
    
    # If no results or error, no graph needed
    if not query_result or query_result == "No results found." or state.get("error"):
        state["needs_graph"] = False
        state["graph_type"] = ""
        return state
    
    prompt = f"""Analyze the following question and query results to determine if a graph visualization would be helpful.

Question: {question}

Query Results Sample:
{query_result[:500]}...

Determine:
1. Would a graph be helpful for this data? (YES/NO)
2. If yes, what type of graph? (bar, line, pie, scatter)

Consider:
- Trends over time → line chart
- Comparisons between categories → bar chart
- Proportions/percentages → pie chart
- Correlations → scatter plot
- Simple counts or single values → NO graph needed

Respond in JSON format:
{{"needs_graph": true/false, "graph_type": "bar/line/pie/scatter/none", "reason": "brief explanation"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data visualization expert. Analyze queries and determine if visualization would add value."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    decision = json.loads(response.choices[0].message.content)
    state["needs_graph"] = decision.get("needs_graph", False)
    state["graph_type"] = decision.get("graph_type", "none")
    
    return state
def viz_agent(state: AgentState) -> AgentState:
    """Generate a graph visualization from query results using LLM-generated Plotly code"""
    query_result = state.get("query_result", "")
    graph_type = state.get("graph_type", "")
    question = state.get("question", "")
    
    try:
        # Parse query results
        query_result_json = state.get("query_result_json", "")
        results = json.loads(query_result_json)
        if not results or len(results) == 0:
            state["graph_json"] = ""
            return state
        
        # Convert to DataFrame for context
        df = pd.DataFrame(results)
        columns = df.columns.tolist()
        sample_data = df.head(5).to_dict('records')
        
        # Generate Plotly code using LLM
        prompt = f"""Generate Python code using Plotly to visualize the following data.

Question: {question}
Graph Type: {graph_type}
Columns: {columns}
Sample Data (first 5 rows): {json.dumps(sample_data, indent=2)}
Total Rows: {len(df)}

Requirements:
1. Use plotly.graph_objects or plotly.express
2. The data is already loaded as 'df' (a pandas DataFrame)
3. Create an appropriate {graph_type} chart
4. Limit data to top 20 rows if there are many rows
5. Add proper titles, labels, and formatting
6. The figure variable must be named 'fig'
7. Return ONLY the Python code, no explanations or markdown
8. Do NOT include any import statements
9. Do NOT include code to show the figure (no fig.show())
10. Make the visualization visually appealing with appropriate colors and layout
11. Update the layout for better interactivity (hover info, responsive sizing)

Generate the Plotly code:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AGENT_CONFIG["viz_agent"]["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        plotly_code = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        plotly_code = plotly_code.replace("```python", "").replace("```", "").strip()
        
        # Prepare execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'json': json
        }
        
        # Import plotly dynamically
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            exec_globals['go'] = go
            exec_globals['px'] = px
        except ImportError:
            print("Plotly not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'plotly'])
            import plotly.graph_objects as go
            import plotly.express as px
            exec_globals['go'] = go
            exec_globals['px'] = px
        
        # Execute the generated code
        exec(plotly_code, exec_globals)
        
        # Get the figure object
        fig = exec_globals.get('fig')
        
        if fig is None:
            raise ValueError("Generated code did not create a 'fig' variable")
        
        # Export figure as JSON for Chainlit's Plotly element
        graph_json = fig.to_json()
        state["graph_json"] = graph_json
        
    except Exception as e:
        print(f"Graph generation error: {e}")
        print(f"Generated code:\n{plotly_code if 'plotly_code' in locals() else 'No code generated'}")
        state["graph_json"] = ""
    
    return state




# -------------------------------
def create_branch_worlds_from_plan(state: AgentState) -> AgentState:
    plan = state.get("branch_plan") or {}
    branches = plan.get("branches", [])

    # plan 안의 branch_id -> world_id 매핑 만들어두기
    branch_to_world: Dict[str, str] = {}

    for branch in branches:
        branch_id = branch.get("branch_id")
        name = branch.get("name", "")
        if not branch_id:
            continue

        desc = f"{branch_id}: {name}"
        world_id = branch_manager.create_world(parent_id="main", description=desc)
        branch_to_world[branch_id] = world_id

        # AgentState.worlds도 업데이트
        state["worlds"][world_id] = branch_manager.get_worlds()[world_id]

    state["branch_plan"]["branch_to_world"] = branch_to_world
    return state
def execute_sql_for_world(state: AgentState, world_id: str, sql: str) -> AgentState:
    result = branch_manager.run_sql(world_id, sql)

    branch_results = state.get("branch_results") or {}
    world_res = branch_results.get(world_id) or {"sql_log": [], "metrics": {}, "samples": []}
    world_res["sql_log"].append(result)
    # metrics/samples는 Evaluate agent에서 따로 정리
    branch_results[world_id] = world_res
    state["branch_results"] = branch_results

    return state
def _mark_world_failed(state: "AgentState", world_id: str, reason: str) -> None:
    """
    내부 헬퍼:
    - 해당 world를 실패 처리하고 failed_worlds, branch_results에 마킹.
    """
    failed_worlds: List[str] = state.get("failed_worlds") or []
    if world_id not in failed_worlds:
        failed_worlds.append(world_id)
    state["failed_worlds"] = failed_worlds

    # branch_results에도 상태 기록
    branch_results: Dict[str, Dict[str, Any]] = state.get("branch_results") or {}
    world_res = branch_results.get(world_id) or {
        "sql_log": [],
        "samples": [],
        "metrics": {},
    }
    world_res["status"] = "failed"
    world_res["failure_reason"] = reason
    branch_results[world_id] = world_res
    state["branch_results"] = branch_results
def _extract_choice_index_from_query(query: str) -> int | None:
    """
    유저가 입력한 문자열에서 1 이상의 정수 하나를 찾아서 반환.
    예:
      - "2번이 좋아" → 2
      - "전략 3으로 해줘" → 3
    못 찾으면 None.
    """
    m = re.search(r"(\d+)", query)
    if not m:
        return None
    try:
        idx = int(m.group(1))
        return idx if idx >= 1 else None
    except ValueError:
        return None
def _strip_code_fences(sql: str) -> str:
    sql = sql.strip()
    # 앞쪽 ```... 제거
    if sql.startswith("```"):
        # 첫 번째 ``` 떼기
        sql = sql.split("```", 1)[1]
        sql = sql.lstrip()
        # ```sql 같은 경우 처리
        if sql.lower().startswith("sql"):
            sql = sql[3:]
        # 뒤쪽 ``` 떼기
        if "```" in sql:
            sql = sql.rsplit("```", 1)[0]
    return sql.strip()
def _extract_metrics_from_world_res(world_res: Dict[str, Any]) -> Dict[str, float]:
    sql_log = world_res.get("sql_log") or []
    last_select = None
    for entry in reversed(sql_log):
        if entry.get("type") == "select":
            last_select = entry
            break
    if not last_select:
        return {}

    rows = last_select.get("rows") or []
    if not rows:
        return {}

    metrics: Dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and value is not None:
                metrics[key] = metrics.get(key, 0.0) + float(value)
    return metrics


def should_generate_graph(state: AgentState) -> str:
    """Decide whether to generate a graph"""
    if state.get("needs_graph", False):
        return "viz_agent"
    return "skip_graph"

# -------------------------------
# BUILD LANGGRAPH
# -------------------------------
def create_text2sql_graph():

    graph = StateGraph(AgentState)

    graph.add_node("guardrails_agent", guardrails_agent)
    graph.add_node("router_agent", router_agent)
    graph.add_node("experiment_planner_agent", experiment_planner_agent)
    graph.add_node("branch_world_creator_agent", branch_world_creator_agent)
    graph.add_node("sql_agent_experiment", sql_agent_experiment)
    graph.add_node("execute_sql_agent", execute_sql_agent)
    graph.add_node("error_agent", error_agent)
    graph.add_node("evaluate_agent", evaluate_agent) 
    graph.add_node("auto_commit_best_world_agent", auto_commit_best_world_agent)
    
    graph.add_node("sql_agent_read_only", sql_agent_read_only)
    graph.add_node("execute_sql_read_only_agent", execute_sql_read_only_agent)
    graph.add_node("sql_agent_schema", sql_agent_schema)

    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("decide_graph_need", decide_graph_need)
    graph.add_node("viz_agent", viz_agent)
    
    # Conditional edge for graph generation
    
    graph.set_entry_point("guardrails_agent")
    graph.add_edge("guardrails_agent", "router_agent")
    graph.add_conditional_edges(
        "router_agent",
        lambda s: s.get("intent"),
        {
            "OUT_OF_SCOPE": END,
            "READ_ONLY": "sql_agent_read_only",      # 예시
            "SCHEMA_CHANGE": "sql_agent_schema",     # 예시
            "EXPERIMENT_START": "experiment_planner_agent",
        },
    )
    graph.add_edge("sql_agent_read_only", "execute_sql_read_only_agent")
    graph.add_edge("execute_sql_read_only_agent", "analysis_agent")
    graph.add_edge("analysis_agent", "decide_graph_need")
    graph.add_conditional_edges(
        "decide_graph_need",
        should_generate_graph,
        {
            "viz_agent": "viz_agent",
            "skip_graph": END
        }
    )
    graph.add_edge("viz_agent", END)

    
    graph.add_edge("sql_agent_schema", "execute_sql_agent")

    graph.add_edge("experiment_planner_agent", "branch_world_creator_agent")
    graph.add_edge("branch_world_creator_agent", "sql_agent_experiment")
    graph.add_edge("sql_agent_experiment", "execute_sql_agent")

    graph.add_conditional_edges(
        "execute_sql_agent",
        lambda state: state.get("needs_error_handling", False),
        {
            True: "error_agent",       # 에러 → SQL 수정 시도
            False: "evaluate_agent",   # 에러 없음 → 평가 단계로
        },
    )
    graph.add_edge("error_agent", "execute_sql_agent") # error_agent 실행 후에는 항상 다시 execute_sql_agent로
    # graph.add_edge("evaluate_agent", END)
    
    # 다음 턴에 2번이 좋아 → BRANCH_CONTROL로 commit”

    # auto_commit_best_world_agent 이후는 chainlit 쪽에서 state를 읽고 응답으로 끝내면 됨.
    
    graph.add_edge("evaluate_agent", "auto_commit_best_world_agent") # error_agent 실행 후에는 항상 다시 execute_sql_agent로
    graph.add_edge("auto_commit_best_world_agent", END)


    return graph.compile()


text2sql_graph = create_text2sql_graph()

def generate_graph_visualization(output_path: str = "text2sql_workflow.png") -> str:
    """
    Generate a PNG visualization of the LangGraph workflow.
    
    Args:
        output_path: Path where the PNG file will be saved (default: "text2sql_workflow.png")
    
    Returns:
        str: Path to the generated PNG file
    """
    try:
        # Get the graph visualization
        graph_image = text2sql_graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(graph_image)
        
        print(f"Graph visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating graph visualization: {e}")
        print("Make sure you have 'pygraphviz' or 'grandalf' installed:")
        print("  pip install pygraphviz")
        print("  or")
        print("  pip install grandalf")
        return None



async def process_question_stream(question: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream node execution events for Chainlit visualization.
    E-commerce multi-branch / agentic speculation workflow 전용.

    - input: 사용자 자연어 질문 (question)
    - output: node_start / node_end / final 이벤트를 순차적으로 yield
    """

    # LangGraph에 넘길 초기 상태 구성
    initial_state: AgentState = {
        # 원 질문
        "question": question,
        "final_answer": "",

        # LangGraph 노드들이 쓰는 공통 필드
        "messages": [{"role": "user", "content": question}],
        "user_query": question,
        "schema_info": SCHEMA_INFO,  # 이 모듈 어딘가에 정의돼 있다고 가정

        # world/branch 관련 초기값
        "branch_sql": {},
        "branch_results": {},
        "branch_sql_progress": {},
        "failed_worlds": [],
        "error_retry_counts": {},

        # guardrail 기본값
        "guardrail_in_scope": True,
        "guardrail_reason": "",
        # intent는 router_agent에서 결정
    }
    initial_state.update(branch_manager.init_state_for_agent())

    # current_state는 계속 업데이트해가면서 Chainlit에 보여줄 스냅샷
    current_state: AgentState = initial_state.copy()

    # LangGraph에서 사용할 실제 노드 이름들 (graph.add_node 할 때 쓴 이름과 동일해야 함)
    tracked_nodes = [
        "guardrails_agent",
        "router_agent",

        "experiment_planner_agent",
        "branch_world_creator_agent",
        "sql_agent_experiment",
        "execute_sql_agent",
        "error_agent",
        "evaluate_agent",

        "sql_agent_read_only",
        "execute_sql_read_only_agent",
        "analysis_agent",
        "decide_graph_need",
        "viz_agent",
        "auto_commit_best_world_agent",
    ]

    try:
        # LangGraph의 이벤트 스트림 구독
        async for event in text2sql_graph.astream_events(
            initial_state,
            config={"recursion_limit": 50},
            version="v1",
        ):
            event_type = event.get("event")
            node_name = event.get("name")

            # 노드 시작
            if event_type == "on_chain_start" and node_name in tracked_nodes:
                yield {
                    "type": "node_start",
                    "node": node_name,
                    "input": current_state.copy(),
                }

            # 노드 종료
            elif event_type == "on_chain_end" and node_name in tracked_nodes:
                output = event.get("data", {}).get("output", {}) or {}

                # LangGraph 노드 함수가 반환한 partial state를 current_state에 반영
                if isinstance(output, dict):
                    current_state.update(output)  # AgentState는 그냥 dict이므로 update OK

                yield {
                    "type": "node_end",
                    "node": node_name,
                    "output": output,
                    "state": current_state.copy(),
                }

        # 그래프 전체 실행 종료
        yield {
            "type": "final",
            "result": current_state,
        }

    except Exception as e:
        # 에러 발생 시
        yield {
            "type": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    # Test the agent
    print("=" * 80)
    print("Text2SQL Agent - Use 'chainlit run app.py' to start the web interface")
    print("=" * 80)
    print("\nThis module is meant to be imported and used via the Chainlit app.")
    print("Run: chainlit run app.py")
