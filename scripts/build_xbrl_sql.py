#!/usr/bin/env python3
import argparse
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


NS = {
    "xsd": "http://www.w3.org/2001/XMLSchema",
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
    "xbrli": "http://www.xbrl.org/2003/instance",
    "xbrldi": "http://xbrl.org/2006/xbrldi",
    "ix": "http://www.xbrl.org/2013/inlineXBRL",
}

XLINK = "{http://www.w3.org/1999/xlink}"
XBRLI = "{http://www.xbrl.org/2003/instance}"
XBRLDI = "{http://xbrl.org/2006/xbrldi}"

ROLE_LABEL = "http://www.xbrl.org/2003/role/label"

INDUSTRY_TYPES = ("CI", "BASI", "FH", "INS", "BD", "MIM")
REPORT_SCOPES = ("CR", "SR")


def split_tag(tag: str) -> Tuple[Optional[str], str]:
    if tag.startswith("{"):
        namespace, local_name = tag[1:].split("}", 1)
        return namespace, local_name
    return None, tag


def parse_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def parse_namespaces(path: Path) -> Dict[str, str]:
    namespaces: Dict[str, str] = {}
    for _, pair in ET.iterparse(path, events=("start-ns",)):
        prefix, uri = pair
        namespaces[prefix or ""] = uri
    return namespaces


def href_to_concept(href: Optional[str]) -> Optional[str]:
    if not href or "#" not in href:
        return None
    return href.split("#", 1)[1]


def concept_prefix(concept_name: str) -> Optional[str]:
    if "_" not in concept_name:
        return None
    return concept_name.split("_", 1)[0]


def infer_family_from_path(path: Path, root_dir: Path) -> str:
    try:
        relative = path.relative_to(root_dir)
    except ValueError:
        relative = path
    parts = [part for part in relative.parts if part and part != "."]
    return parts[0] if parts else "root"


def resolve_href_path(base_file: Path, href: str, root_dir: Path) -> str:
    href_file = href.split("#", 1)[0]
    if not href_file:
        return normalize_path_for_sql(base_file, root_dir) or str(base_file)
    if href_file.startswith("http://") or href_file.startswith("https://"):
        return href
    return normalize_path_for_sql((base_file.parent / href_file).resolve(), root_dir) or href


def find_taxonomy_version(text: str) -> Optional[str]:
    match = re.search(r"(tifrs-\d{8})", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def sanitize_token(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_") or "unknown"


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return f"'{value.isoformat()}'"
    text = str(value).replace("'", "''")
    return f"'{text}'"


def insert_sql(table: str, row: Dict, replace: bool = True) -> str:
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    columns = ", ".join(row.keys())
    values = ", ".join(sql_literal(value) for value in row.values())
    return f"{verb} INTO {table} ({columns}) VALUES ({values});"


def normalize_path_for_sql(path: Optional[Path], root_dir: Optional[Path] = None) -> Optional[str]:
    if path is None:
        return None
    if root_dir is None:
        return str(path)
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def safe_int(value: Optional[str]) -> Optional[int]:
    if value in (None, "", "INF", "-INF", "NaN"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def safe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def discover_instance_files(root_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xbrl", ".xml", ".html", ".htm"}
    )


def dedupe_instance_paths(paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    seen = {}
    skipped: List[Path] = []
    for path in sorted(paths):
        report_id = path.stem
        if report_id in seen:
            skipped.append(path)
            continue
        seen[report_id] = path
    return list(seen.values()), skipped


def parse_date(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def infer_quarter(period_end: Optional[date]) -> Optional[str]:
    if period_end is None:
        return None
    if period_end.month <= 3:
        return "Q1"
    if period_end.month <= 6:
        return "Q2"
    if period_end.month <= 9:
        return "Q3"
    return "Q4"


def infer_statement_type(role_uri: Optional[str], module: Optional[str]) -> str:
    role_uri = role_uri or ""
    if "BalanceSheet" in role_uri:
        return "balance_sheet"
    if "ComprehensiveIncome" in role_uri or "ProfitLoss" in role_uri:
        return "comprehensive_income_statement"
    if "CashFlows" in role_uri:
        return "statement_of_cash_flows"
    if module == "SCF":
        return "statement_of_cash_flows"
    return "disclosure"


def qname_to_concept_id(
    qname_text: Optional[str],
    namespaces: Dict[str, str],
    concept_lookup: Dict[Tuple[Optional[str], str], str],
) -> Optional[str]:
    if not qname_text:
        return None
    if ":" in qname_text:
        prefix, local_name = qname_text.split(":", 1)
        namespace = namespaces.get(prefix)
        if namespace is not None:
            return concept_lookup.get((namespace, local_name)) or f"{prefix}_{local_name}"
        return qname_text
    return concept_lookup.get((None, qname_text)) or qname_text


@dataclass
class ConceptRecord:
    concept_id: str
    taxonomy_id: str
    namespace: Optional[str]
    local_name: str
    zh_label: Optional[str] = None
    en_label: Optional[str] = None
    terse_code: Optional[str] = None
    period_type: Optional[str] = None
    balance_type: Optional[str] = None
    data_type: Optional[str] = None
    is_abstract: Optional[int] = None
    roles: set = field(default_factory=set)
    source_files: set = field(default_factory=set)


@dataclass
class TaxonomyParseResult:
    version: Optional[str]
    entry_points: Dict[str, Dict]
    concepts: Dict[str, ConceptRecord]
    concept_lookup: Dict[Tuple[Optional[str], str], str]
    presentation_rows: List[Dict]
    calculation_rows: List[Dict]


@dataclass
class ContextInfo:
    context_id: str
    entity_identifier: Optional[str]
    instant_date: Optional[date]
    period_start: Optional[date]
    period_end: Optional[date]
    segment_json: Optional[str]


def discover_taxonomy_files(root_dir: Path) -> Dict[str, List[Path]]:
    categories = {
        "xsd": [],
        "label": [],
        "presentation": [],
        "calculation": [],
    }
    for path in sorted(p for p in root_dir.rglob("*") if p.is_file()):
        lower_name = path.name.lower()
        suffix = path.suffix.lower()
        if lower_name == ".ds_store":
            continue
        if suffix == ".xsd":
            categories["xsd"].append(path)
        elif suffix == ".xml" and "label" in lower_name:
            categories["label"].append(path)
        elif suffix == ".xml" and "presentation" in lower_name:
            categories["presentation"].append(path)
        elif suffix == ".xml" and "calculation" in lower_name:
            categories["calculation"].append(path)
    return categories


def build_taxonomy_id(version: Optional[str], family: str) -> str:
    prefix = version or "taxonomy"
    return f"{prefix}:{sanitize_token(family)}"


def ensure_entry_point(entry_points: Dict[str, Dict], taxonomy_id: str, version: Optional[str], module: str) -> Dict:
    item = entry_points.setdefault(
        taxonomy_id,
        {
            "taxonomy_id": taxonomy_id,
            "taxonomy_version": version,
            "module": module,
            "entry_points": set(),
            "xsd_files": set(),
            "presentation_files": set(),
            "label_files": set(),
            "calculation_files": set(),
        },
    )
    return item


def parse_entry_points(root_dir: Path, xsd_files: Iterable[Path], version: Optional[str]) -> Dict[str, Dict]:
    entry_points: Dict[str, Dict] = {}
    for xsd_path in xsd_files:
        root = parse_xml(xsd_path)
        family = infer_family_from_path(xsd_path, root_dir)
        taxonomy_id = build_taxonomy_id(version, family)
        item = ensure_entry_point(entry_points, taxonomy_id, version, family)
        item["xsd_files"].add(normalize_path_for_sql(xsd_path, root_dir))

        has_linkbase_ref = False
        for linkbase_ref in root.findall(".//link:linkbaseRef", NS):
            has_linkbase_ref = True
            href = linkbase_ref.attrib.get(f"{XLINK}href")
            role = (linkbase_ref.attrib.get(f"{XLINK}role") or "").lower()
            if href:
                item["entry_points"].add(normalize_path_for_sql(xsd_path, root_dir))
                resolved_href = resolve_href_path(xsd_path, href, root_dir)
                if "presentation" in role or "presentation" in href.lower():
                    item["presentation_files"].add(resolved_href)
                elif "label" in role or "label" in href.lower():
                    item["label_files"].add(resolved_href)
                elif "calculation" in role or "calculation" in href.lower():
                    item["calculation_files"].add(resolved_href)

        if not has_linkbase_ref:
            item["entry_points"].add(normalize_path_for_sql(xsd_path, root_dir))

    return entry_points


def parse_taxonomy_xsds(
    root_dir: Path,
    xsd_files: Iterable[Path],
    version: Optional[str],
    entry_points: Dict[str, Dict],
) -> Tuple[Dict[str, ConceptRecord], Dict[Tuple[Optional[str], str], str]]:
    concepts: Dict[str, ConceptRecord] = {}
    concept_lookup: Dict[Tuple[Optional[str], str], str] = {}

    for xsd_path in xsd_files:
        root = parse_xml(xsd_path)
        family = infer_family_from_path(xsd_path, root_dir)
        taxonomy_id = build_taxonomy_id(version, family)
        ensure_entry_point(entry_points, taxonomy_id, version, family)["xsd_files"].add(
            normalize_path_for_sql(xsd_path, root_dir)
        )

        target_namespace = root.attrib.get("targetNamespace")
        for elem in root.findall("xsd:element", NS):
            local_name = elem.attrib.get("name")
            concept_id = elem.attrib.get("id") or local_name
            if not local_name or not concept_id:
                continue
            item = concepts.get(concept_id)
            if item is None:
                item = ConceptRecord(
                    concept_id=concept_id,
                    taxonomy_id=taxonomy_id,
                    namespace=target_namespace,
                    local_name=local_name,
                )
                concepts[concept_id] = item
            item.namespace = item.namespace or target_namespace
            item.local_name = item.local_name or local_name
            item.taxonomy_id = item.taxonomy_id or taxonomy_id
            item.period_type = item.period_type or elem.attrib.get(f"{XBRLI}periodType")
            item.balance_type = item.balance_type or elem.attrib.get(f"{XBRLI}balance")
            item.data_type = item.data_type or elem.attrib.get("type")
            abstract_value = elem.attrib.get("abstract")
            if item.is_abstract is None and abstract_value is not None:
                item.is_abstract = 1 if abstract_value.lower() == "true" else 0
            item.source_files.add(normalize_path_for_sql(xsd_path, root_dir))
            concept_lookup[(target_namespace, local_name)] = concept_id
    return concepts, concept_lookup


def parse_label_files(
    root_dir: Path,
    label_files: Iterable[Path],
    concepts: Dict[str, ConceptRecord],
    version: Optional[str],
    entry_points: Dict[str, Dict],
) -> None:
    for label_path in label_files:
        root = parse_xml(label_path)
        family = infer_family_from_path(label_path, root_dir)
        taxonomy_id = build_taxonomy_id(version, family)
        ensure_entry_point(entry_points, taxonomy_id, version, family)["label_files"].add(
            normalize_path_for_sql(label_path, root_dir)
        )

        for label_link in root.findall("link:labelLink", NS):
            locators = {}
            labels = {}
            for loc in label_link.findall("link:loc", NS):
                locators[loc.attrib.get(f"{XLINK}label")] = loc.attrib.get(f"{XLINK}href")
            for label in label_link.findall("link:label", NS):
                labels[label.attrib.get(f"{XLINK}label")] = {
                    "role": label.attrib.get(f"{XLINK}role"),
                    "lang": label.attrib.get("{http://www.w3.org/XML/1998/namespace}lang"),
                    "text": (label.text or "").strip(),
                }
            for arc in label_link.findall("link:labelArc", NS):
                concept_id = href_to_concept(locators.get(arc.attrib.get(f"{XLINK}from")))
                resource = labels.get(arc.attrib.get(f"{XLINK}to"))
                if not concept_id or not resource or concept_id not in concepts:
                    continue
                item = concepts[concept_id]
                item.source_files.add(normalize_path_for_sql(label_path, root_dir))
                if resource["role"] == ROLE_LABEL and resource["lang"] == "zh-tw" and resource["text"]:
                    item.zh_label = resource["text"]
                elif resource["role"] == ROLE_LABEL and resource["lang"] == "en" and resource["text"]:
                    item.en_label = resource["text"]
                elif "terseLabel" in (resource["role"] or "") and resource["lang"] == "en" and resource["text"]:
                    item.terse_code = resource["text"]


def parse_presentation_files(
    root_dir: Path,
    presentation_files: Iterable[Path],
    concepts: Dict[str, ConceptRecord],
    version: Optional[str],
    entry_points: Dict[str, Dict],
) -> List[Dict]:
    rows: List[Dict] = []
    for presentation_path in presentation_files:
        root = parse_xml(presentation_path)
        family = infer_family_from_path(presentation_path, root_dir)
        taxonomy_id = build_taxonomy_id(version, family)
        ensure_entry_point(entry_points, taxonomy_id, version, family)["presentation_files"].add(
            normalize_path_for_sql(presentation_path, root_dir)
        )

        for link in root.findall("link:presentationLink", NS):
            role_uri = link.attrib.get(f"{XLINK}role")
            locators = {
                loc.attrib.get(f"{XLINK}label"): loc.attrib.get(f"{XLINK}href")
                for loc in link.findall("link:loc", NS)
            }
            children_by_parent = defaultdict(list)
            parents = set()

            for arc in link.findall("link:presentationArc", NS):
                parent_id = href_to_concept(locators.get(arc.attrib.get(f"{XLINK}from")))
                child_id = href_to_concept(locators.get(arc.attrib.get(f"{XLINK}to")))
                if not parent_id or not child_id:
                    continue
                if parent_id in concepts:
                    concepts[parent_id].roles.add(role_uri)
                if child_id in concepts:
                    concepts[child_id].roles.add(role_uri)
                children_by_parent[parent_id].append(
                    (
                        child_id,
                        safe_float(arc.attrib.get("order")) or 0.0,
                        arc.attrib.get("preferredLabel"),
                    )
                )
                parents.add(child_id)

            roots = [node for node in children_by_parent if node not in parents]

            def walk(parent_id: str, depth: int) -> None:
                for child_id, order_no, preferred_label in sorted(children_by_parent.get(parent_id, []), key=lambda item: item[1]):
                    rows.append(
                        {
                            "taxonomy_id": taxonomy_id,
                            "role_uri": role_uri,
                            "parent_concept_id": parent_id,
                            "child_concept_id": child_id,
                            "order_no": order_no,
                            "preferred_label": preferred_label,
                            "depth": depth,
                        }
                    )
                    walk(child_id, depth + 1)

            for root_id in roots:
                walk(root_id, 1)
    return rows


def parse_calculation_files(
    root_dir: Path,
    calculation_files: Iterable[Path],
    concepts: Dict[str, ConceptRecord],
    version: Optional[str],
    entry_points: Dict[str, Dict],
) -> List[Dict]:
    rows: List[Dict] = []
    for calculation_path in calculation_files:
        root = parse_xml(calculation_path)
        family = infer_family_from_path(calculation_path, root_dir)
        taxonomy_id = build_taxonomy_id(version, family)
        ensure_entry_point(entry_points, taxonomy_id, version, family)["calculation_files"].add(
            normalize_path_for_sql(calculation_path, root_dir)
        )

        for link in root.findall("link:calculationLink", NS):
            role_uri = link.attrib.get(f"{XLINK}role")
            locators = {
                loc.attrib.get(f"{XLINK}label"): loc.attrib.get(f"{XLINK}href")
                for loc in link.findall("link:loc", NS)
            }
            for arc in link.findall("link:calculationArc", NS):
                parent_id = href_to_concept(locators.get(arc.attrib.get(f"{XLINK}from")))
                child_id = href_to_concept(locators.get(arc.attrib.get(f"{XLINK}to")))
                if not parent_id or not child_id:
                    continue
                if parent_id in concepts:
                    concepts[parent_id].roles.add(role_uri)
                if child_id in concepts:
                    concepts[child_id].roles.add(role_uri)
                rows.append(
                    {
                        "taxonomy_id": taxonomy_id,
                        "role_uri": role_uri,
                        "parent_concept_id": parent_id,
                        "child_concept_id": child_id,
                        "weight": safe_float(arc.attrib.get("weight")) or 1.0,
                        "order_no": safe_float(arc.attrib.get("order")) or 0.0,
                    }
                )
    return rows


def parse_taxonomy(root_dir: Path) -> TaxonomyParseResult:
    categories = discover_taxonomy_files(root_dir)
    version = find_taxonomy_version(str(root_dir))
    if version is None:
        for path in categories["xsd"]:
            version = find_taxonomy_version(str(path))
            if version:
                break

    entry_points = parse_entry_points(root_dir, categories["xsd"], version)
    concepts, concept_lookup = parse_taxonomy_xsds(root_dir, categories["xsd"], version, entry_points)
    parse_label_files(root_dir, categories["label"], concepts, version, entry_points)
    presentation_rows = parse_presentation_files(root_dir, categories["presentation"], concepts, version, entry_points)
    calculation_rows = parse_calculation_files(root_dir, categories["calculation"], concepts, version, entry_points)

    return TaxonomyParseResult(
        version=version,
        entry_points=entry_points,
        concepts=concepts,
        concept_lookup=concept_lookup,
        presentation_rows=presentation_rows,
        calculation_rows=calculation_rows,
    )


def parse_context(
    context_elem: ET.Element,
    namespaces: Dict[str, str],
    concept_lookup: Dict[Tuple[Optional[str], str], str],
) -> ContextInfo:
    entity_identifier = None
    entity_elem = context_elem.find("xbrli:entity/xbrli:identifier", NS)
    if entity_elem is not None and entity_elem.text:
        entity_identifier = entity_elem.text.strip()

    instant_date = None
    period_start = None
    period_end = None
    period_elem = context_elem.find("xbrli:period", NS)
    if period_elem is not None:
        instant_date = parse_date((period_elem.findtext("xbrli:instant", default=None, namespaces=NS) or "").strip() or None)
        period_start = parse_date((period_elem.findtext("xbrli:startDate", default=None, namespaces=NS) or "").strip() or None)
        period_end = parse_date((period_elem.findtext("xbrli:endDate", default=None, namespaces=NS) or "").strip() or None)

    segment_members = []
    for member in context_elem.findall(".//xbrldi:explicitMember", NS):
        dimension = member.attrib.get("dimension")
        member_qname = (member.text or "").strip()
        dimension_id = qname_to_concept_id(dimension, namespaces, concept_lookup) or dimension
        member_id = qname_to_concept_id(member_qname, namespaces, concept_lookup) or member_qname
        segment_members.append(
            {
                "type": "explicitMember",
                "dimension": dimension_id,
                "member": member_id,
            }
        )

    segment_json = json.dumps(segment_members, ensure_ascii=False) if segment_members else None
    return ContextInfo(
        context_id=context_elem.attrib["id"],
        entity_identifier=entity_identifier,
        instant_date=instant_date,
        period_start=period_start,
        period_end=period_end,
        segment_json=segment_json,
    )


def parse_unit(unit_elem: ET.Element) -> str:
    measures = [(measure.text or "").strip() for measure in unit_elem.findall("xbrli:measure", NS) if (measure.text or "").strip()]
    if measures:
        return "|".join(measures)
    divide_elem = unit_elem.find("xbrli:divide", NS)
    if divide_elem is not None:
        numerator = [
            (measure.text or "").strip()
            for measure in divide_elem.findall("xbrli:unitNumerator/xbrli:measure", NS)
            if (measure.text or "").strip()
        ]
        denominator = [
            (measure.text or "").strip()
            for measure in divide_elem.findall("xbrli:unitDenominator/xbrli:measure", NS)
            if (measure.text or "").strip()
        ]
        return f"{'|'.join(numerator)}/{'|'.join(denominator)}"
    return ""


def coerce_numeric(value_text: str) -> Optional[Decimal]:
    candidate = value_text.strip().replace(",", "")
    if not candidate or candidate.lower() in {"nil", "none", "nan"}:
        return None
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def apply_scale(numeric_value: Optional[Decimal], scale: Optional[int], sign: Optional[str]) -> Optional[Decimal]:
    if numeric_value is None:
        return None
    result = numeric_value
    if scale:
        result = result * (Decimal(10) ** scale)
    if sign == "-":
        result = result * Decimal(-1)
    return result


def infer_industry_type(*values: Optional[str]) -> Optional[str]:
    haystack = " ".join(value or "" for value in values).upper()
    for code in INDUSTRY_TYPES:
        if re.search(rf"(^|[^A-Z]){code}([^A-Z]|$)", haystack):
            return code
    return None


def infer_report_scope(*values: Optional[str]) -> Optional[str]:
    haystack = " ".join(value or "" for value in values).upper()
    for code in REPORT_SCOPES:
        if re.search(rf"(^|[^A-Z]){code}([^A-Z]|$)", haystack):
            return code
    return None


def infer_module(
    schema_refs: List[str],
    facts: List[Dict],
    concepts: Dict[str, ConceptRecord],
) -> Optional[str]:
    for href in schema_refs:
        upper_href = href.upper()
        for module in ("BSCI", "SCF", "ES", "NOTES"):
            if module in upper_href:
                return module

    counts = Counter()
    for fact in facts:
        concept = concepts.get(fact["concept_id"])
        if concept:
            module = concept.taxonomy_id.split(":")[-1]
            counts[module] += 1
    return counts.most_common(1)[0][0] if counts else None


def parse_instance(
    instance_path: Path,
    taxonomy: TaxonomyParseResult,
) -> Tuple[Dict, List[Dict]]:
    root = parse_xml(instance_path)
    namespaces = parse_namespaces(instance_path)

    contexts = {
        context.attrib["id"]: parse_context(context, namespaces, taxonomy.concept_lookup)
        for context in root.findall(".//xbrli:context", NS)
    }
    units = {
        unit.attrib["id"]: parse_unit(unit)
        for unit in root.findall(".//xbrli:unit", NS)
    }

    schema_refs = []
    for elem in root.iter():
        _, local_name = split_tag(elem.tag)
        if local_name == "schemaRef":
            href = elem.attrib.get(f"{XLINK}href")
            if href:
                schema_refs.append(href)

    facts: List[Dict] = []
    report_period_end = None
    report_period_start = None
    identifier = None
    for index, elem in enumerate(root.iter(), start=1):
        namespace, local_name = split_tag(elem.tag)
        if local_name in {"html", "head", "body", "xbrl", "context", "unit", "schemaRef", "footnoteLink", "resources", "references", "header", "hidden"}:
            continue
        context_id = elem.attrib.get("contextRef")
        if not context_id:
            continue

        fact_name = elem.attrib.get("name")
        concept_id = None
        if fact_name:
            concept_id = qname_to_concept_id(fact_name, namespaces, taxonomy.concept_lookup)
        elif namespace != NS["ix"]:
            concept_id = taxonomy.concept_lookup.get((namespace, local_name))
            if concept_id is None:
                default_prefix = next((prefix for prefix, uri in namespaces.items() if uri == namespace and prefix), None)
                if default_prefix:
                    concept_id = f"{default_prefix}_{local_name}"
        if concept_id is None:
            continue

        context_info = contexts.get(context_id)
        if context_info:
            report_period_end = report_period_end or context_info.period_end or context_info.instant_date
            report_period_start = report_period_start or context_info.period_start
            identifier = identifier or context_info.entity_identifier

        raw_text = (elem.text or "").strip()
        scale = safe_int(elem.attrib.get("scale"))
        sign = elem.attrib.get("sign")
        numeric_value = apply_scale(coerce_numeric(raw_text), scale, sign)
        unit_ref = elem.attrib.get("unitRef")
        if unit_ref is None and numeric_value is not None:
            unit_ref = "pure"
        facts.append(
            {
                "fact_id": f"{instance_path.stem}:{index}",
                "concept_id": concept_id,
                "context_id": context_id,
                "unit_id": unit_ref,
                "value_numeric": float(numeric_value) if numeric_value is not None else None,
                "value_text": None if numeric_value is not None else raw_text or None,
                "decimals": safe_int(elem.attrib.get("decimals")),
                "scale": scale,
                "instant_date": context_info.instant_date.isoformat() if context_info and context_info.instant_date else None,
                "period_start": context_info.period_start.isoformat() if context_info and context_info.period_start else None,
                "period_end": context_info.period_end.isoformat() if context_info and context_info.period_end else None,
                "segment_json": context_info.segment_json if context_info else None,
                "unit_text": units.get(unit_ref),
            }
        )

    taxonomy_version = taxonomy.version
    for href in schema_refs:
        taxonomy_version = taxonomy_version or find_taxonomy_version(href)

    module = infer_module(schema_refs, facts, taxonomy.concepts)
    report_id = instance_path.stem
    report_row = {
        "report_id": report_id,
        "company_code": identifier,
        "year": report_period_end.year if report_period_end else None,
        "quarter": infer_quarter(report_period_end),
        "industry_type": infer_industry_type(report_id, " ".join(schema_refs)),
        "report_scope": infer_report_scope(report_id, " ".join(schema_refs)),
        "taxonomy_version": taxonomy_version,
        "module": module,
        "file_name": instance_path.name,
        "source_type": instance_path.suffix.lstrip(".").lower(),
        "period_start": report_period_start.isoformat() if report_period_start else None,
        "period_end": report_period_end.isoformat() if report_period_end else None,
    }

    for fact in facts:
        fact["report_id"] = report_id

    return report_row, facts


def dedupe_rows(rows: List[Dict], keys: List[str]) -> List[Dict]:
    seen = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def load_sql_into_db(db_path: Path, sql_text: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql_text)
        conn.commit()
    finally:
        conn.close()


def build_auto_field_rows(
    report_row: Dict,
    facts: List[Dict],
    concepts: Dict[str, ConceptRecord],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    field_rows: Dict[str, Dict] = {}
    mapping_rows: Dict[Tuple[str, str, str], Dict] = {}
    metric_rows: List[Dict] = []

    for fact in facts:
        concept = concepts.get(fact["concept_id"])
        if concept is None:
            continue
        field_id = concept.concept_id
        if field_id not in field_rows:
            statement_type = "disclosure"
            if concept.roles:
                statement_type = infer_statement_type(sorted(concept.roles)[0], concept.taxonomy_id.split(":")[-1])
            field_rows[field_id] = {
                "field_id": field_id,
                "canonical_name": concept.local_name,
                "zh_name": concept.zh_label,
                "en_name": concept.en_label,
                "module": concept.taxonomy_id.split(":")[-1],
                "statement_type": statement_type,
                "value_type": "numeric" if fact["value_numeric"] is not None else "text",
                "description": f"Auto-generated 1:1 field mapping for concept {concept.concept_id}",
            }

        mapping_key = (field_id, concept.concept_id, concept.taxonomy_id)
        if mapping_key not in mapping_rows:
            mapping_rows[mapping_key] = {
                "field_id": field_id,
                "concept_id": concept.concept_id,
                "taxonomy_id": concept.taxonomy_id,
                "industry_type": report_row["industry_type"],
                "priority": 1,
                "effective_from": report_row["period_start"],
                "effective_to": report_row["period_end"],
            }

        metric_rows.append(
            {
                "company_code": report_row["company_code"],
                "year": report_row["year"],
                "quarter": report_row["quarter"],
                "report_scope": report_row["report_scope"],
                "industry_type": report_row["industry_type"],
                "field_id": field_id,
                "concept_id": concept.concept_id,
                "value": fact["value_numeric"],
                "report_id": report_row["report_id"],
                "fact_id": fact["fact_id"],
            }
        )

    return list(field_rows.values()), list(mapping_rows.values()), metric_rows


def load_custom_field_mapping(path: Path) -> Tuple[List[Dict], List[Dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("field mapping JSON must be an object with 'fields' and 'mappings'.")
    fields = payload.get("fields", [])
    mappings = payload.get("mappings", [])
    if not isinstance(fields, list) or not isinstance(mappings, list):
        raise ValueError("'fields' and 'mappings' must both be arrays.")
    return fields, mappings


def build_custom_metric_rows(
    report_row: Dict,
    facts: List[Dict],
    mappings: List[Dict],
) -> List[Dict]:
    numeric_facts_by_concept = defaultdict(list)
    for fact in facts:
        numeric_facts_by_concept[fact["concept_id"]].append(fact)

    metric_rows: List[Dict] = []
    for mapping in mappings:
        concept_ids = mapping.get("concept_ids")
        if concept_ids is None and mapping.get("concept_id"):
            concept_ids = [mapping["concept_id"]]
        if not concept_ids:
            continue
        for concept_id in concept_ids:
            for fact in numeric_facts_by_concept.get(concept_id, []):
                metric_rows.append(
                    {
                        "company_code": report_row["company_code"],
                        "year": report_row["year"],
                        "quarter": report_row["quarter"],
                        "report_scope": report_row["report_scope"],
                        "industry_type": report_row["industry_type"],
                        "field_id": mapping["field_id"],
                        "concept_id": concept_id,
                        "value": fact["value_numeric"],
                        "report_id": report_row["report_id"],
                        "fact_id": fact["fact_id"],
                    }
                )
    return metric_rows


def render_sql(
    taxonomy: TaxonomyParseResult,
    report_row: Optional[Dict],
    facts: List[Dict],
    field_rows: List[Dict],
    concept_mapping_rows: List[Dict],
    metric_rows: List[Dict],
    include_taxonomy: bool = True,
) -> str:
    lines = ["BEGIN TRANSACTION;"]

    if include_taxonomy:
        for entry in sorted(taxonomy.entry_points.values(), key=lambda item: item["taxonomy_id"]):
            lines.append(
                insert_sql(
                    "taxonomy_entry_point",
                    {
                        "taxonomy_id": entry["taxonomy_id"],
                        "taxonomy_version": entry["taxonomy_version"],
                        "module": entry["module"],
                        "entry_point": "|".join(sorted(entry["entry_points"])) or None,
                        "xsd_file": "|".join(sorted(entry["xsd_files"])) or None,
                        "presentation_file": "|".join(sorted(entry["presentation_files"])) or None,
                        "label_file": "|".join(sorted(entry["label_files"])) or None,
                        "calculation_file": "|".join(sorted(entry["calculation_files"])) or None,
                    },
                )
            )

        for concept in sorted(taxonomy.concepts.values(), key=lambda item: item.concept_id):
            lines.append(
                insert_sql(
                    "taxonomy_concept",
                    {
                        "concept_id": concept.concept_id,
                        "taxonomy_id": concept.taxonomy_id,
                        "namespace": concept.namespace,
                        "local_name": concept.local_name,
                        "zh_label": concept.zh_label,
                        "en_label": concept.en_label,
                        "terse_code": concept.terse_code,
                        "period_type": concept.period_type,
                        "balance_type": concept.balance_type,
                        "data_type": concept.data_type,
                        "is_abstract": concept.is_abstract,
                    },
                )
            )

        for row in taxonomy.presentation_rows:
            lines.append(insert_sql("taxonomy_presentation", row, replace=False))

        for row in taxonomy.calculation_rows:
            lines.append(insert_sql("taxonomy_calculation", row, replace=False))

    if report_row is not None:
        lines.append(insert_sql("report_instance", report_row))

    for fact in facts:
        lines.append(
            insert_sql(
                "xbrl_fact",
                {
                    "fact_id": fact["fact_id"],
                    "report_id": fact["report_id"],
                    "concept_id": fact["concept_id"],
                    "context_id": fact["context_id"],
                    "unit_id": fact["unit_id"] or fact["unit_text"],
                    "value_numeric": fact["value_numeric"],
                    "value_text": fact["value_text"],
                    "decimals": fact["decimals"],
                    "scale": fact["scale"],
                    "instant_date": fact["instant_date"],
                    "period_start": fact["period_start"],
                    "period_end": fact["period_end"],
                    "segment_json": fact["segment_json"],
                },
            )
        )

    for row in field_rows:
        lines.append(insert_sql("field_dictionary", row))

    for row in concept_mapping_rows:
        lines.append(insert_sql("field_concept_mapping", row, replace=False))

    for row in metric_rows:
        lines.append(insert_sql("financial_metric_value", row, replace=False))

    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse tifrs-20200630 taxonomy and an XBRL instance into SQL INSERT statements."
    )
    parser.add_argument("--taxonomy-root", required=True, help="Root directory of the XBRL taxonomy, e.g. tifrs-20200630.")
    parser.add_argument("--instance", help="Path to the XBRL instance file to parse.")
    parser.add_argument("--instance-dir", help="Recursively parse all XBRL/iXBRL files under this directory.")
    parser.add_argument("--sql-output", required=True, help="Output path for generated SQL.")
    parser.add_argument("--db-path", help="Optional SQLite DB path. When set, generated SQL is loaded into the DB.")
    parser.add_argument(
        "--field-mode",
        choices=("auto-concept", "custom"),
        default="auto-concept",
        help="auto-concept creates a 1:1 field<->concept mapping; custom loads field definitions from JSON.",
    )
    parser.add_argument(
        "--field-mapping-json",
        help="When --field-mode custom is used, load {fields: [...], mappings: [...]} from this JSON file.",
    )
    args = parser.parse_args()

    taxonomy_root = Path(args.taxonomy_root).resolve()
    sql_output = Path(args.sql_output).resolve()

    taxonomy = parse_taxonomy(taxonomy_root)

    report_row = None
    facts: List[Dict] = []
    field_rows: List[Dict] = []
    concept_mapping_rows: List[Dict] = []
    metric_rows: List[Dict] = []

    report_rows: List[Dict] = []
    all_facts: List[Dict] = []
    all_field_rows: List[Dict] = []
    all_concept_mapping_rows: List[Dict] = []
    all_metric_rows: List[Dict] = []

    instance_paths: List[Path] = []
    if args.instance:
        instance_paths.append(Path(args.instance).resolve())
    if args.instance_dir:
        instance_paths.extend(discover_instance_files(Path(args.instance_dir).resolve()))
    instance_paths = sorted(dict.fromkeys(instance_paths))
    instance_paths, skipped_instance_paths = dedupe_instance_paths(instance_paths)

    raw_mapping_rows: List[Dict] = []
    if args.field_mode == "custom":
        if not args.field_mapping_json:
            raise SystemExit("--field-mapping-json is required when --field-mode custom is used.")
        field_rows, raw_mapping_rows = load_custom_field_mapping(Path(args.field_mapping_json).resolve())
        all_field_rows.extend(field_rows)

    for instance_path in instance_paths:
        report_row, facts = parse_instance(instance_path, taxonomy)
        report_rows.append(report_row)
        all_facts.extend(facts)

        if args.field_mode == "custom":
            for mapping in raw_mapping_rows:
                concept_ids = mapping.get("concept_ids")
                if concept_ids is None and mapping.get("concept_id"):
                    concept_ids = [mapping["concept_id"]]
                for concept_id in concept_ids or []:
                    all_concept_mapping_rows.append(
                        {
                            "field_id": mapping["field_id"],
                            "concept_id": concept_id,
                            "taxonomy_id": mapping.get("taxonomy_id"),
                            "industry_type": mapping.get("industry_type"),
                            "priority": mapping.get("priority", 1),
                            "effective_from": mapping.get("effective_from"),
                            "effective_to": mapping.get("effective_to"),
                        }
                    )
            all_metric_rows.extend(build_custom_metric_rows(report_row, facts, raw_mapping_rows))
        else:
            current_field_rows, current_concept_mapping_rows, current_metric_rows = build_auto_field_rows(report_row, facts, taxonomy.concepts)
            all_field_rows.extend(current_field_rows)
            all_concept_mapping_rows.extend(current_concept_mapping_rows)
            all_metric_rows.extend(current_metric_rows)

    all_field_rows = dedupe_rows(all_field_rows, ["field_id"])
    all_concept_mapping_rows = dedupe_rows(
        all_concept_mapping_rows,
        ["field_id", "concept_id", "taxonomy_id", "industry_type", "effective_from", "effective_to"],
    )

    sql_blocks = [render_sql(taxonomy, None, [], [], [], [], include_taxonomy=True)]
    for report_row in report_rows:
        report_facts = [fact for fact in all_facts if fact["report_id"] == report_row["report_id"]]
        report_metrics = [row for row in all_metric_rows if row["report_id"] == report_row["report_id"]]
        sql_blocks.append(render_sql(taxonomy, report_row, report_facts, [], [], report_metrics, include_taxonomy=False))

    if all_field_rows or all_concept_mapping_rows:
        lines = ["BEGIN TRANSACTION;"]
        for row in all_field_rows:
            lines.append(insert_sql("field_dictionary", row))
        for row in all_concept_mapping_rows:
            lines.append(insert_sql("field_concept_mapping", row, replace=False))
        lines.append("COMMIT;")
        sql_blocks.append("\n".join(lines) + "\n")

    sql_text = "".join(sql_blocks)
    sql_output.parent.mkdir(parents=True, exist_ok=True)
    sql_output.write_text(sql_text, encoding="utf-8")

    if args.db_path:
        load_sql_into_db(Path(args.db_path).resolve(), sql_text)

    print(f"taxonomy_version={taxonomy.version}")
    print(f"taxonomy_entry_points={len(taxonomy.entry_points)}")
    print(f"taxonomy_concepts={len(taxonomy.concepts)}")
    print(f"taxonomy_presentation_rows={len(taxonomy.presentation_rows)}")
    print(f"taxonomy_calculation_rows={len(taxonomy.calculation_rows)}")
    print(f"reports={len(report_rows)}")
    print(f"skipped_duplicate_reports={len(skipped_instance_paths)}")
    print(f"facts={len(all_facts)}")
    print(f"field_dictionary_rows={len(all_field_rows)}")
    print(f"field_concept_mapping_rows={len(all_concept_mapping_rows)}")
    print(f"financial_metric_value_rows={len(all_metric_rows)}")
    print(f"sql_output={sql_output}")


if __name__ == "__main__":
    main()
