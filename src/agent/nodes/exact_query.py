import json
import logging
import sqlite3
from typing import Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.mappings.company_stock_code_array import CompanyStockCodeArray
from src.providers.chat_openAI_provider import chat_model
from src.services.account_title_matcher import find_candidates
from src.types.langgraph_state_types import OverallState


logger = logging.getLogger(__name__)
DB_PATH = "FinancialStatementXBRL.db"
VALID_STATEMENT_TYPES = {
    "balance_sheet",
    "comprehensive_income_statement",
    "statement_of_cash_flows",
}
METADATA_FIELD_PREFIXES = (
    "tifrs-notes_Company",
    "tifrs-notes_Year",
    "tifrs-notes_Quarter",
    "tifrs-notes_Report",
    "tifrs-notes_Market",
    "tifrs-notes_Industry",
)


class Period(BaseModel):
    year: int = Field(..., description="使用者提問中提到的年度")
    quarter: int = Field(..., description="使用者提問中提到的季度，Q1 只回傳 1")
    range: Optional[str] = Field(None, description="季度期間文字，例如 2024年Q1到Q3")


class RequestedField(BaseModel):
    field: str = Field(..., description="使用者請求的會計項目名稱，例如 現金及約當現金")
    category: Optional[str] = Field(
        None,
        description="項目所屬的報表類別，例如 資產負債表、綜合損益表、現金流量表",
    )


class QuestionSchema(BaseModel):
    companyName: str = Field(description="使用者提問中提到的公司全名")
    companyCode: str = Field(description="公司股票代碼")
    shortName: str = Field(description="公司常用簡稱")
    englishName: str = Field(description="公司英文名稱簡寫")
    period: Period
    requested_fields: List[RequestedField]


def dump_log_payload(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_company_maps() -> tuple[Dict[str, Dict], Dict[str, Dict]]:
    code_to_company_map = {}
    name_to_company_map = {}
    for item in CompanyStockCodeArray:
        code_to_company_map[item["companyCode"]] = item
        for key in ["companyName", "shortName", "englishName"]:
            value = item.get(key)
            if value:
                name_to_company_map[value] = item
    return code_to_company_map, name_to_company_map


def resolve_company(schema: Dict) -> Dict:
    code_to_company_map, name_to_company_map = build_company_maps()
    company_identifiers = [
        schema.get("companyCode"),
        schema.get("companyName"),
        schema.get("shortName"),
        schema.get("englishName"),
    ]

    for index, identifier in enumerate(company_identifiers):
        if not identifier:
            continue
        found_company = (
            code_to_company_map.get(identifier)
            if index == 0
            else name_to_company_map.get(identifier)
        )
        if found_company is None:
            matches = [
                item
                for item in CompanyStockCodeArray
                if any(
                    isinstance(value, str) and identifier in value
                    for value in item.values()
                )
            ]
            found_company = matches[0] if matches else None
        if found_company:
            schema["companyName"] = found_company["companyName"]
            schema["companyCode"] = found_company["companyCode"]
            schema["shortName"] = found_company["shortName"]
            schema["englishName"] = found_company["englishName"]
            return schema
    return schema


def extract_question_schema(question: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=QuestionSchema)
    prompt = PromptTemplate(
        template="""盡可能回覆問題，並組成指定格式。無法取得資訊的欄位填入空字串。
companyName 一定要從問題中取出文字代入。
quarter 請回傳 1 到 4 的整數。

問題：{question}
格式：{format_instructions}""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | chat_model | parser
    return chain.invoke({"question": question})


def filter_candidates(candidates: List[Dict]) -> List[Dict]:
    filtered = []
    for candidate in candidates:
        concept_name = candidate.get("concept_name") or ""
        if any(concept_name.startswith(prefix) for prefix in METADATA_FIELD_PREFIXES):
            continue
        filtered.append(candidate)
    return filtered


def select_candidate(
    user_question: str,
    field_name: str,
    statement_type: str,
    candidates: List[Dict],
) -> Optional[Dict]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    options = [
        {
            "concept_name": item.get("concept_name"),
            "code": item.get("code"),
            "zh_tw": item.get("zh_tw"),
            "en": item.get("en"),
            "score": item.get("score"),
        }
        for item in candidates
    ]

    prompt = f"""
你是一個財報欄位對應助手。
請依照使用者問題，從候選清單中選出最符合的 concept_name。
只能從候選清單中挑選。
只回答一個 concept_name，不要解釋。

### 報表別
{statement_type}

### 使用者問題
{user_question}

### 使用者要找的欄位
{field_name}

### 候選清單
{options}
"""
    response = chat_model.invoke(prompt)
    concept_name = str(response.content).strip()
    for candidate in candidates:
        if candidate.get("concept_name") == concept_name:
            return candidate
    return candidates[0]


def fetch_financial_value(
    company_code: str,
    year: int,
    quarter: int,
    statement_type: str,
    concept_id: str,
) -> Optional[Dict[str, str]]:
    if statement_type not in VALID_STATEMENT_TYPES:
        return None

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            SELECT
                ri.report_id,
                ri.company_code,
                ri.year,
                ri.quarter,
                ri.report_scope,
                ri.industry_type,
                ri.period_end AS report_period_end,
                fd.field_id,
                fd.canonical_name,
                fd.zh_name,
                fd.en_name,
                fd.statement_type,
                fmv.concept_id,
                fmv.value,
                xf.value_numeric,
                xf.value_text,
                xf.unit_id,
                xf.instant_date,
                xf.period_start,
                xf.period_end,
                xf.segment_json
            FROM financial_metric_value AS fmv
            JOIN report_instance AS ri
              ON ri.report_id = fmv.report_id
            LEFT JOIN field_dictionary AS fd
              ON fd.field_id = fmv.field_id
            LEFT JOIN xbrl_fact AS xf
              ON xf.fact_id = fmv.fact_id
            WHERE ri.company_code = ?
              AND ri.year = ?
              AND ri.quarter = ?
              AND fmv.concept_id = ?
              AND fmv.value IS NOT NULL
              AND (fd.statement_type = ? OR fd.statement_type IS NULL)
              AND (
                    (fd.statement_type = 'balance_sheet' AND xf.instant_date = ri.period_end)
                 OR (fd.statement_type IN ('comprehensive_income_statement', 'statement_of_cash_flows') AND xf.period_end = ri.period_end)
                 OR (fd.statement_type IS NULL AND (xf.instant_date = ri.period_end OR xf.period_end = ri.period_end))
              )
            ORDER BY
                CASE WHEN xf.segment_json IS NULL THEN 0 ELSE 1 END,
                CASE WHEN xf.unit_id = 'TWD' THEN 0 ELSE 1 END,
                ABS(fmv.value) DESC
            LIMIT 1
            """,
            (company_code, year, f"Q{quarter}", concept_id, statement_type),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        connection.close()


def resolve_answer_data(
    schema: Dict,
    statement_type: str,
    selected_candidate: Dict,
    candidates: List[Dict],
) -> tuple[Optional[Dict], Dict, List[Dict]]:
    ordered_candidates = [selected_candidate] + [
        item for item in candidates if item.get("concept_name") != selected_candidate.get("concept_name")
    ]
    attempt_logs: List[Dict] = []
    for candidate in ordered_candidates:
        answer_data = fetch_financial_value(
            company_code=schema["companyCode"],
            year=schema["period"]["year"],
            quarter=schema["period"]["quarter"],
            statement_type=statement_type,
            concept_id=candidate["concept_name"],
        )
        attempt_logs.append(
            {
                "concept_name": candidate.get("concept_name"),
                "code": candidate.get("code"),
                "zh_tw": candidate.get("zh_tw"),
                "en": candidate.get("en"),
                "found": answer_data is not None,
                "report_id": answer_data.get("report_id") if answer_data else None,
                "value": answer_data.get("value") if answer_data else None,
                "unit_id": answer_data.get("unit_id") if answer_data else None,
                "period_end": answer_data.get("period_end") if answer_data else None,
                "instant_date": answer_data.get("instant_date") if answer_data else None,
            }
        )
        if answer_data:
            return answer_data, candidate, attempt_logs
    return None, selected_candidate, attempt_logs


def exact_query(state: OverallState) -> OverallState:
    logger.info("[exact_query] input state:\n%s", dump_log_payload(state))
    schema = extract_question_schema(state["user_input"])
    schema = resolve_company(schema)

    requested_fields = schema.get("requested_fields", [])
    if not requested_fields:
        return {
            **state,
            "answer": "無法從問題中辨識要查詢的財務欄位。",
        }

    target_field = requested_fields[0]["field"]
    statement_type = state.get("statement_type", "")
    candidates = filter_candidates(find_candidates(target_field, statement_type, limit=8))

    if not candidates:
        return {
            **state,
            "answer": f"找不到與「{target_field}」對應的資料字典欄位。",
            "reference_data": {"schema": schema, "candidates": []},
        }

    selected_candidate = select_candidate(
        user_question=state["user_input"],
        field_name=target_field,
        statement_type=statement_type,
        candidates=candidates,
    )
    if not selected_candidate:
        return {
            **state,
            "answer": f"無法判斷「{target_field}」對應的財報欄位。",
            "reference_data": {"schema": schema, "candidates": candidates},
        }

    answer_data, resolved_candidate, attempt_logs = resolve_answer_data(
        schema=schema,
        statement_type=statement_type,
        selected_candidate=selected_candidate,
        candidates=candidates,
    )

    debug_payload = {
        "query_context": {
            "company_code": schema["companyCode"],
            "company_name": schema["companyName"],
            "year": schema["period"]["year"],
            "quarter": schema["period"]["quarter"],
            "statement_type": statement_type,
            "target_field": target_field,
            "selected_candidate": {
                "concept_name": selected_candidate.get("concept_name"),
                "code": selected_candidate.get("code"),
                "zh_tw": selected_candidate.get("zh_tw"),
                "en": selected_candidate.get("en"),
            },
            "candidate_count": len(candidates),
        },
        "candidates": [
            {
                "concept_name": candidate.get("concept_name"),
                "code": candidate.get("code"),
                "zh_tw": candidate.get("zh_tw"),
                "en": candidate.get("en"),
                "score": candidate.get("score"),
            }
            for candidate in candidates
        ],
        "attempts": attempt_logs,
    }
    logger.info("[exact_query] lookup debug:\n%s", dump_log_payload(debug_payload))

    if not answer_data:
        return {
            **state,
            "answer": (
                f"已比對到欄位 {selected_candidate['concept_name']}"
                f"（{selected_candidate.get('zh_tw') or selected_candidate.get('en') or target_field}），"
                f"但查無 {schema['companyName']} {schema['period']['year']} 年 "
                f"Q{schema['period']['quarter']} 的主期間資料。"
            ),
            "reference_data": {
                "schema": schema,
                "selected_candidate": selected_candidate,
                "candidates": candidates,
            },
        }

    final_prompt = f"""
你是一個專業的信用徵審團隊助手，請根據資料庫查到的財務報表資料直接回答問題。
若答案為數字，請保留正負號，加入千分位格式，並帶出單位。
若該欄位中文名稱存在，優先用中文欄位名稱表達。
不要臆測，僅根據提供資料回答。

### 問題
{state['user_input']}

### 查詢條件
公司：{schema['companyName']} ({schema['companyCode']})
期間：{schema['period']['year']} 年 Q{schema['period']['quarter']}
報表別：{statement_type}
使用者欄位：{target_field}

### 欄位對應
concept_id：{selected_candidate['concept_name']}
code：{resolved_candidate.get('code')}
中文名稱：{resolved_candidate.get('zh_tw')}
英文名稱：{resolved_candidate.get('en')}

### 財務報表資料
{answer_data}
"""

    final_answer = chat_model.invoke(final_prompt).content
    return {
        **state,
        "answer": final_answer,
        "reference_data": {
            "schema": schema,
            "selected_candidate": resolved_candidate,
            "candidates": candidates,
            "answer_data": answer_data,
        },
    }
