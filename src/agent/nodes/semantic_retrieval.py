import json
import logging
import sqlite3
from typing import Dict, List, Literal, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator

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


class PeriodItem(BaseModel):
    year: int = Field(..., description="西元年")
    quarter: int = Field(..., description="季度，1 到 4")


class RequirementDraft(BaseModel):
    field_query: List[str] = Field(
        default_factory=list,
        description="要查的財務欄位陣列，例如 ['營業收入', '營收']、['資產總額']、['本期淨利', '稅後淨利']",
    )
    statement_type: str = Field(
        ...,
        description="只可填 balance_sheet、comprehensive_income_statement、statement_of_cash_flows",
    )
    periods: List[PeriodItem] = Field(default_factory=list, description="此欄位需要查的期間")
    purpose: str = Field(..., description="查這些數據是為了回答什麼，例如比較、趨勢、計算差異")

    @field_validator("field_query", mode="before")
    @classmethod
    def normalize_field_query(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class SemanticPlanDraft(BaseModel):
    company_identifier: str = Field(..., description="公司代碼、公司全名、簡稱或英文名")
    analysis_goal: str = Field(..., description="對問題的高層理解，例如比較營收趨勢、分析獲利變化")
    requirements: List[RequirementDraft] = Field(default_factory=list, description="回答所需的資料清單")


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


def resolve_company(identifier: str) -> Optional[Dict]:
    if not identifier:
        return None

    code_to_company_map, name_to_company_map = build_company_maps()
    direct = code_to_company_map.get(identifier) or name_to_company_map.get(identifier)
    if direct:
        return direct

    for item in CompanyStockCodeArray:
        if any(
            isinstance(value, str) and identifier in value
            for value in item.values()
        ):
            return item
    return None


def list_company_reports(company_code: str) -> List[Dict]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            SELECT report_id, company_code, year, quarter, report_scope, industry_type, module, period_end
            FROM report_instance
            WHERE company_code = ?
            ORDER BY year DESC, quarter DESC
            """,
            (company_code,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def filter_candidates(candidates: List[Dict]) -> List[Dict]:
    filtered = []
    for candidate in candidates:
        concept_name = candidate.get("concept_name") or ""
        if any(concept_name.startswith(prefix) for prefix in METADATA_FIELD_PREFIXES):
            continue
        filtered.append(candidate)
    return filtered


def dedupe_candidates(candidates: List[Dict], limit: int) -> List[Dict]:
    seen = {}
    for candidate in candidates:
        key = candidate.get("concept_name")
        if key and key not in seen:
            seen[key] = candidate
    items = list(seen.values())
    items.sort(
        key=lambda item: (
            -(item.get("score") or 0),
            item.get("statement_type") or "",
            item.get("code") or "",
            item.get("concept_name") or "",
        )
    )
    return items[:limit]


def search_candidates_across_statements(field_queries: List[str], statement_type: str, limit: int) -> List[Dict]:
    target_types = (
        [statement_type]
        if statement_type in VALID_STATEMENT_TYPES
        else sorted(VALID_STATEMENT_TYPES)
    )
    collected: List[Dict] = []
    normalized_queries = [query.strip() for query in field_queries if isinstance(query, str) and query.strip()]
    for current_statement_type in target_types:
        for field_query in normalized_queries:
            for item in filter_candidates(find_candidates(field_query, current_statement_type, limit=limit)):
                enriched = dict(item)
                enriched["statement_type"] = current_statement_type
                enriched["matched_query"] = field_query
                collected.append(enriched)
    return dedupe_candidates(collected, limit)


def fetch_financial_value(
    company_code: str,
    year: int,
    quarter: int,
    statement_type: str,
    concept_id: str,
) -> Optional[Dict]:
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
        return dict(row) if row else None
    finally:
        connection.close()


def extract_semantic_plan(question: str) -> Dict:
    parser = JsonOutputParser(pydantic_object=SemanticPlanDraft)
    prompt = PromptTemplate(
        template="""你是財務資料需求規劃器。
            你的任務是先判斷：要回答使用者問題，至少需要哪些財務數據。

            規則：
            1. company_identifier 一定要填公司代碼、公司名、簡稱或英文名，從問題中擷取。
            2. statement_type 只能填：
            - balance_sheet
            - comprehensive_income_statement
            - statement_of_cash_flows
            3. requirements 要列出回答此題真正需要查的欄位。
            4. 每個 requirement 的 field_query 必須是字串陣列；若有同義詞、近義欄位或複數表達，請全部放進陣列。
            5. periods 只填問題中明確提到、或回答此題必要的期間。
            6. 如果問題需要比較多個期間，就列出多個 periods。
            7. 若沒有辦法判斷，requirements 仍盡量列出最可能需要的欄位。

            問題：{question}
            格式：{format_instructions}""",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    formatted_prompt = prompt.format(question=question)
    response = chat_model.invoke(formatted_prompt)
    print(
        "[semantic_retrieval] extract_semantic_plan raw llm response:\n"
        + dump_log_payload(
            {
                "question": question,
                "response_content": response.content,
            }
        )
    )
    return parser.invoke(response)


def choose_best_candidate(question: str, requirement: Dict, candidates: List[Dict]) -> Dict:
    if not candidates:
        return {}
    if len(candidates) == 1:
        return candidates[0]
    top_candidate = candidates[0]
    second_candidate = candidates[1] if len(candidates) > 1 else None
    top_score = float(top_candidate.get("score") or 0)
    second_score = float(second_candidate.get("score") or 0) if second_candidate else 0.0
    if top_score >= second_score + 12:
        # print(
        #     "[semantic_retrieval] choose_best_candidate fast path:\n"
        #     + json.dumps(
        #         {
        #             "reason": "top score margin is large enough",
        #             "top_candidate": {
        #                 "concept_name": top_candidate.get("concept_name"),
        #                 "zh_tw": top_candidate.get("zh_tw"),
        #                 "en": top_candidate.get("en"),
        #                 "code": top_candidate.get("code"),
        #                 "matched_query": top_candidate.get("matched_query"),
        #                 "score": top_candidate.get("score"),
        #             },
        #             "second_candidate": {
        #                 "concept_name": second_candidate.get("concept_name"),
        #                 "zh_tw": second_candidate.get("zh_tw"),
        #                 "en": second_candidate.get("en"),
        #                 "code": second_candidate.get("code"),
        #                 "matched_query": second_candidate.get("matched_query"),
        #                 "score": second_candidate.get("score"),
        #             } if second_candidate else None,
        #         },
        #         ensure_ascii=False,
        #         indent=2,
        #     )
        # )
        return top_candidate

    compact_candidates = [
        {
            "concept_name": candidate.get("concept_name"),
            "zh_tw": candidate.get("zh_tw"),
            "en": candidate.get("en"),
            "code": candidate.get("code"),
            "statement_type": candidate.get("statement_type"),
            "matched_query": candidate.get("matched_query"),
            "score": candidate.get("score"),
        }
        for candidate in candidates[:3]
    ]
    compact_requirement = {
        "field_query": requirement.get("field_query"),
        "statement_type": requirement.get("statement_type"),
        "periods": requirement.get("periods"),
        "purpose": requirement.get("purpose"),
    }

    prompt = f"""
你是財務欄位選擇器。
請根據使用者問題與資料需求，從候選清單中選出最適合查資料的一個 concept_name。
只能回答 concept_name，不要解釋。

### 使用者問題
{question}

### 資料需求
{json.dumps(compact_requirement, ensure_ascii=False, indent=2)}

### 候選清單
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}
"""
    try:
        response = chat_model.invoke(prompt)
        chosen = str(response.content).strip()
        for candidate in candidates:
            if candidate.get("concept_name") == chosen:
                return candidate
    except Exception as exc:
        print(f"[semantic_retrieval] choose_best_candidate fallback due to error: {exc}")
    return candidates[0]


def build_llm_evidence_candidate(candidate: Dict) -> Dict:
    if not candidate:
        return {}
    return {
        "concept_name": candidate.get("concept_name"),
        "zh_tw": candidate.get("zh_tw"),
        "en": candidate.get("en"),
        "code": candidate.get("code"),
        "statement_type": candidate.get("statement_type"),
        "matched_query": candidate.get("matched_query"),
        "score": candidate.get("score"),
        "mapped_from": candidate.get("mapped_from"),
        "mapping_queries": candidate.get("mapping_queries"),
    }


def retrieve_requirement_data(question: str, company: Dict, requirement: Dict) -> Dict:
    field_queries = requirement.get("field_query", [])
    query_results = []
    values = []

    for field_query in field_queries:
        query_requirement = dict(requirement)
        query_requirement["field_query"] = [field_query]
        candidates = search_candidates_across_statements(
            field_queries=[field_query],
            statement_type=requirement["statement_type"],
            limit=8,
        )
        # print(
        #     "[semantic_retrieval] candidates:\n"
        #     + json.dumps(
        #         {
        #             "requirement": query_requirement,
        #             "field_query": field_query,
        #             "candidates": candidates,
        #         },
        #         ensure_ascii=False,
        #         indent=2,
        #     )
        # )
        selected_candidate = choose_best_candidate(question, query_requirement, candidates)

        query_values = []
        for period in requirement.get("periods", []):
            result = None
            if selected_candidate:
                result = fetch_financial_value(
                    company_code=company["companyCode"],
                    year=period["year"],
                    quarter=period["quarter"],
                    statement_type=selected_candidate.get("statement_type") or requirement["statement_type"],
                    concept_id=selected_candidate.get("concept_name"),
                )
            value_item = {
                "field_query": field_query,
                "period": period,
                "result": result,
            }
            query_values.append(value_item)
            values.append(value_item)

        query_results.append(
            {
                "field_query": field_query,
                "selected_candidate": build_llm_evidence_candidate(selected_candidate),
                "candidates": [build_llm_evidence_candidate(candidate) for candidate in candidates],
                "values": query_values,
            }
        )

    return {
        "requirement": requirement,
        "query_results": query_results,
        "values": values,
    }


def semantic_retrieval(state: OverallState) -> OverallState:
    print("semantic_retrieval in =======")

    question = state["rephrased_question"] or state["user_input"]
    try:
        plan = extract_semantic_plan(question)
        print(f"[semantic_retrieval] extract_semantic_plan plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}")
    except Exception as exc:
        return {
            **state,
            "answer": f"語意檢索規劃階段失敗，暫時無法分析所需財務資料。錯誤：{exc}",
            "reference_data": {"question": question},
        }
    # print("\n********** [semantic_retrieval] AI AGENT data-requirement plan start **********")
    # print(json.dumps(plan, ensure_ascii=False, indent=2))
    # print("********** [semantic_retrieval] AI AGENT data-requirement plan end **********\n")

    company = resolve_company(plan.get("company_identifier", ""))
    if not company:
        return {
            **state,
            "answer": "無法辨識問題中的公司，因此無法進一步查詢資料庫。",
            "reference_data": {"plan": plan},
        }

    available_reports = list_company_reports(company["companyCode"])
    # print("\n[semantic_retrieval] company:")
    # print(json.dumps(company, ensure_ascii=False, indent=2))
    # print("\n[semantic_retrieval] available_reports:")
    # print(json.dumps(available_reports[:20], ensure_ascii=False, indent=2))

    retrieval_results = []
    for requirement in plan.get("requirements", []):
        result = retrieve_requirement_data(question, company, requirement)
        retrieval_results.append(result)

    evidence_json = {
        "question": question,
        "analysis_goal": plan.get("analysis_goal"),
        "company": company,
        "available_reports": available_reports,
        "retrieval_results": retrieval_results,
    }
    # print("\n[semantic_retrieval] evidence_json:")
    # print(json.dumps(evidence_json, ensure_ascii=False, indent=2))

    fulfilled_items = 0
    planned_items = 0
    fulfilled_details = []
    planned_details = []
    for item in retrieval_results:
        requirement = item.get("requirement", {})
        query_result_map = {
            query_result.get("field_query"): query_result
            for query_result in item.get("query_results", [])
        }
        for value_item in item["values"]:
            field_query = value_item.get("field_query")
            query_result = query_result_map.get(field_query, {})
            detail = {
                "requirement": requirement,
                "field_query": field_query,
                "selected_candidate": query_result.get("selected_candidate", {}),
                "period": value_item.get("period"),
                "result": value_item.get("result"),
                "is_fulfilled": value_item.get("result") is not None,
            }
            planned_details.append(detail)
            planned_items += 1
            if value_item.get("result") is not None:
                fulfilled_items += 1
                fulfilled_details.append(detail)

    # print(
    #     "[semantic_retrieval] planned_details:\n"
    #     + json.dumps(planned_details, ensure_ascii=False, indent=2, default=str)
    # )
    # print(
    #     "[semantic_retrieval] fulfilled_details:\n"
    #     + json.dumps(fulfilled_details, ensure_ascii=False, indent=2, default=str)
    # )

    print("fulfilled_items =", fulfilled_items)
    print("planned_items =", planned_items)
    enough_information = fulfilled_items > 0 
    # enough_information = fulfilled_items > 0 and fulfilled_items == planned_items

    # print(
    #     json.dumps(
    #         {
    #             "planned_items": planned_items,
    #             "fulfilled_items": fulfilled_items,
    #             "enough_information": enough_information,
    #         },
    #         ensure_ascii=False,
    #         indent=2,
    #     )
    # )

    if not enough_information:
        return {
            **state,
            "answer": "我已分析需要的財務資料並查詢資料庫，但目前資料不足以完整回答這個問題。",
            "reference_data": evidence_json,
        }

    final_prompt = f"""
        你是信用徵審財報分析助手。
        請根據下列 JSON 證據資料回答問題。

        規則：
        1. 只能根據 JSON 中已查到的資料回答，不要自行臆測。
        2. 若答案需要比較、趨勢、增減、成長率，請直接用 JSON 中的數值計算或描述。
        3. 最終回答請用繁體中文。
        4. 數值請加上千分位，並盡量帶單位。

        ### 使用者問題
        {question}

        ### JSON 證據資料
        {json.dumps(evidence_json, ensure_ascii=False, indent=2)}
        """
    try:
        final_answer = chat_model.invoke(final_prompt).content
    except Exception as exc:
        final_answer = (
            "已查到足夠的財務資料，但最終分析回答階段失敗。"
            f"你可以先參考 reference_data 中的 JSON 證據。錯誤：{exc}"
        )
    # print("\n[semantic_retrieval] final_answer:")
    # print(final_answer)

    return {
        **state,
        "answer": final_answer,
        "reference_data": evidence_json,
    }
