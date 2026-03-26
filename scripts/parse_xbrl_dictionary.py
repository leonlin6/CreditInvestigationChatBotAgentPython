#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import xml.etree.ElementTree as ET


NS = {
    "xsd": "http://www.w3.org/2001/XMLSchema",
    "link": "http://www.xbrl.org/2003/linkbase",
    "xlink": "http://www.w3.org/1999/xlink",
    "xbrli": "http://www.xbrl.org/2003/instance",
}

XLINK = "{http://www.w3.org/1999/xlink}"
XBRLI = "{http://www.xbrl.org/2003/instance}"

LABEL_ROLE = "http://www.xbrl.org/2003/role/label"
TERSE_ROLE = "http://www.xbrl.org/2003/role/terseLabel"
TOTAL_ROLE = "http://www.xbrl.org/2003/role/totalLabel"


def href_to_concept(href: Optional[str]) -> Optional[str]:
    if not href or "#" not in href:
        return None
    return href.split("#", 1)[1]


def concept_prefix(concept_name: str) -> Optional[str]:
    if not concept_name or "_" not in concept_name:
        return None
    return concept_name.split("_", 1)[0]


def infer_family_from_path(path: Path, root_dir: Path) -> str:
    try:
        relative = path.relative_to(root_dir)
    except ValueError:
        relative = path
    parts = [part for part in relative.parts if part and part != "."]
    return parts[0] if parts else "root"


def is_local_href(href: Optional[str], root_dir: Path) -> bool:
    if not href or href.startswith("http://") or href.startswith("https://"):
        return False
    file_part = href.split("#", 1)[0]
    if not file_part:
        return True
    return (root_dir / file_part).exists()


def ensure_concept(
    concepts: Dict[str, dict],
    concept_name: str,
    root_dir: Path,
    href: Optional[str] = None,
    source_file: Optional[Path] = None,
) -> dict:
    item = concepts.setdefault(
        concept_name,
        {
            "concept_name": concept_name,
            "namespace_prefix": concept_prefix(concept_name),
            "name": concept_name.split("_", 1)[1] if "_" in concept_name else concept_name,
            "id": concept_name,
            "type": None,
            "substitution_group": None,
            "abstract": None,
            "nillable": None,
            "period_type": None,
            "balance": None,
            "source_kind": None,
            "source_hrefs": set(),
            "source_files": set(),
            "families": set(),
            "labels": defaultdict(dict),
            "code": None,
            "presentation": [],
            "calculation_children": [],
            "calculation_parents": [],
            "definition_children": [],
            "definition_parents": [],
            "role_occurrences": set(),
        },
    )
    if href:
        item["source_hrefs"].add(href)
        if item["source_kind"] is None:
            item["source_kind"] = "local" if is_local_href(href, root_dir) else "external"
    if source_file:
        item["source_files"].add(str(source_file))
        item["families"].add(infer_family_from_path(source_file, root_dir))
    return item


def parse_xsd(xsd_path: Path, root_dir: Path, concepts: Dict[str, dict]) -> None:
    root = ET.parse(xsd_path).getroot()
    for elem in root.findall("xsd:element", NS):
        name = elem.attrib.get("name")
        if not name:
            continue
        concept_name = elem.attrib.get("id", name)
        item = ensure_concept(
            concepts,
            concept_name,
            root_dir,
            href=f"{xsd_path}#{concept_name}",
            source_file=xsd_path,
        )
        item.update(
            {
                "name": name,
                "id": concept_name,
                "type": elem.attrib.get("type"),
                "substitution_group": elem.attrib.get("substitutionGroup"),
                "abstract": elem.attrib.get("abstract") == "true",
                "nillable": elem.attrib.get("nillable") == "true",
                "period_type": elem.attrib.get(f"{XBRLI}periodType"),
                "balance": elem.attrib.get(f"{XBRLI}balance"),
                "source_kind": "local",
            }
        )


def parse_label(label_path: Path, root_dir: Path, concepts: Dict[str, dict]) -> None:
    root = ET.parse(label_path).getroot()
    for label_link in root.findall("link:labelLink", NS):
        locators = {}
        resources = {}
        for loc in label_link.findall("link:loc", NS):
            locators[loc.attrib.get(f"{XLINK}label")] = loc.attrib.get(f"{XLINK}href")
        for label in label_link.findall("link:label", NS):
            resources[label.attrib.get(f"{XLINK}label")] = {
                "role": label.attrib.get(f"{XLINK}role"),
                "lang": label.attrib.get("{http://www.w3.org/XML/1998/namespace}lang"),
                "text": (label.text or "").strip(),
            }
        for arc in label_link.findall("link:labelArc", NS):
            from_label = arc.attrib.get(f"{XLINK}from")
            to_label = arc.attrib.get(f"{XLINK}to")
            href = locators.get(from_label)
            resource = resources.get(to_label)
            concept_name = href_to_concept(href)
            if not concept_name or not resource:
                continue
            item = ensure_concept(concepts, concept_name, root_dir, href=href, source_file=label_path)
            role = resource["role"] or "unknown"
            lang = resource["lang"] or "und"
            text = resource["text"]
            if text:
                item["labels"][role][lang] = text


def parse_presentation(presentation_path: Path, root_dir: Path, concepts: Dict[str, dict]) -> None:
    root = ET.parse(presentation_path).getroot()
    for link in root.findall("link:presentationLink", NS):
        role = link.attrib.get(f"{XLINK}role")
        locators = {}
        for loc in link.findall("link:loc", NS):
            locators[loc.attrib.get(f"{XLINK}label")] = loc.attrib.get(f"{XLINK}href")

        parents_by_role = defaultdict(list)
        children_by_role = defaultdict(list)

        for arc in link.findall("link:presentationArc", NS):
            parent_label = arc.attrib.get(f"{XLINK}from")
            child_label = arc.attrib.get(f"{XLINK}to")
            parent_href = locators.get(parent_label)
            child_href = locators.get(child_label)
            parent_concept = href_to_concept(parent_href)
            child_concept = href_to_concept(child_href)
            if not parent_concept or not child_concept:
                continue
            parent_item = ensure_concept(concepts, parent_concept, root_dir, parent_href, presentation_path)
            child_item = ensure_concept(concepts, child_concept, root_dir, child_href, presentation_path)
            parent_item["role_occurrences"].add(role)
            child_item["role_occurrences"].add(role)
            child_item["presentation"].append(
                {
                    "role": role,
                    "parent_concept": parent_concept,
                    "order": float(arc.attrib.get("order", "0")),
                    "preferred_label": arc.attrib.get("preferredLabel"),
                    "source_file": str(presentation_path),
                }
            )
            parents_by_role[parent_concept].append(child_concept)
            children_by_role[child_concept].append(parent_concept)

        roots = [node for node in parents_by_role if node not in children_by_role]

        def walk(node: str, trail: List[str], seen: Set[str]) -> None:
            for child in parents_by_role.get(node, []):
                if child in seen:
                    continue
                child_item = concepts[child]
                path_entry = {
                    "role": role,
                    "path": trail + [child],
                    "source_file": str(presentation_path),
                }
                if path_entry not in child_item["presentation"]:
                    child_item["presentation"].append(path_entry)
                walk(child, trail + [child], seen | {child})

        for root_concept in roots:
            root_item = concepts[root_concept]
            root_path = {
                "role": role,
                "path": [root_concept],
                "source_file": str(presentation_path),
            }
            if root_path not in root_item["presentation"]:
                root_item["presentation"].append(root_path)
            walk(root_concept, [root_concept], {root_concept})


def parse_calculation(calculation_path: Path, root_dir: Path, concepts: Dict[str, dict]) -> None:
    root = ET.parse(calculation_path).getroot()
    for link in root.findall("link:calculationLink", NS):
        role = link.attrib.get(f"{XLINK}role")
        locators = {}
        for loc in link.findall("link:loc", NS):
            locators[loc.attrib.get(f"{XLINK}label")] = loc.attrib.get(f"{XLINK}href")
        for arc in link.findall("link:calculationArc", NS):
            parent_label = arc.attrib.get(f"{XLINK}from")
            child_label = arc.attrib.get(f"{XLINK}to")
            parent_href = locators.get(parent_label)
            child_href = locators.get(child_label)
            parent_concept = href_to_concept(parent_href)
            child_concept = href_to_concept(child_href)
            if not parent_concept or not child_concept:
                continue
            parent_item = ensure_concept(concepts, parent_concept, root_dir, parent_href, calculation_path)
            child_item = ensure_concept(concepts, child_concept, root_dir, child_href, calculation_path)
            parent_item["role_occurrences"].add(role)
            child_item["role_occurrences"].add(role)
            child_record = {
                "role": role,
                "concept_name": child_concept,
                "order": float(arc.attrib.get("order", "0")),
                "weight": float(arc.attrib.get("weight", "1")),
                "source_file": str(calculation_path),
            }
            parent_record = {
                "role": role,
                "concept_name": parent_concept,
                "order": float(arc.attrib.get("order", "0")),
                "weight": float(arc.attrib.get("weight", "1")),
                "source_file": str(calculation_path),
            }
            parent_item["calculation_children"].append(child_record)
            child_item["calculation_parents"].append(parent_record)


def parse_definition(definition_path: Path, root_dir: Path, concepts: Dict[str, dict]) -> None:
    root = ET.parse(definition_path).getroot()
    for link in root.findall("link:definitionLink", NS):
        role = link.attrib.get(f"{XLINK}role")
        locators = {}
        for loc in link.findall("link:loc", NS):
            locators[loc.attrib.get(f"{XLINK}label")] = loc.attrib.get(f"{XLINK}href")
        for arc in link.findall("link:definitionArc", NS):
            parent_label = arc.attrib.get(f"{XLINK}from")
            child_label = arc.attrib.get(f"{XLINK}to")
            parent_href = locators.get(parent_label)
            child_href = locators.get(child_label)
            parent_concept = href_to_concept(parent_href)
            child_concept = href_to_concept(child_href)
            if not parent_concept or not child_concept:
                continue
            arcrole = arc.attrib.get(f"{XLINK}arcrole")
            parent_item = ensure_concept(concepts, parent_concept, root_dir, parent_href, definition_path)
            child_item = ensure_concept(concepts, child_concept, root_dir, child_href, definition_path)
            parent_item["role_occurrences"].add(role)
            child_item["role_occurrences"].add(role)
            child_record = {
                "role": role,
                "arcrole": arcrole,
                "concept_name": child_concept,
                "order": float(arc.attrib.get("order", "0")),
                "source_file": str(definition_path),
            }
            parent_record = {
                "role": role,
                "arcrole": arcrole,
                "concept_name": parent_concept,
                "order": float(arc.attrib.get("order", "0")),
                "source_file": str(definition_path),
            }
            parent_item["definition_children"].append(child_record)
            child_item["definition_parents"].append(parent_record)


def discover_files(root_dir: Path) -> Dict[str, List[Path]]:
    categories = {
        "xsd": [],
        "label": [],
        "presentation": [],
        "calculation": [],
        "definition": [],
    }
    for path in sorted(p for p in root_dir.rglob("*") if p.is_file()):
        name = path.name.lower()
        if name == ".ds_store" or path.suffix.lower() not in {".xsd", ".xml"}:
            continue
        if path.suffix.lower() == ".xsd":
            categories["xsd"].append(path)
        elif "label" in name:
            categories["label"].append(path)
        elif "presentation" in name:
            categories["presentation"].append(path)
        elif "calculation" in name:
            categories["calculation"].append(path)
        elif "definition" in name:
            categories["definition"].append(path)
    return categories


def dedupe_records(records: Iterable[dict], sort_keys: List[str]) -> List[dict]:
    def normalize(value):
        if value is None:
            return ("none", "")
        if isinstance(value, list):
            return ("list", tuple(normalize(v) for v in value))
        if isinstance(value, dict):
            return ("dict", tuple(sorted((k, normalize(v)) for k, v in value.items())))
        if isinstance(value, (int, float)):
            return ("number", value)
        return ("text", str(value))

    seen = {}
    for record in records:
        key = tuple(normalize(record.get(k)) for k in sort_keys)
        if key not in seen:
            seen[key] = record
    return sorted(seen.values(), key=lambda x: tuple(normalize(x.get(k)) for k in sort_keys))


def finalize_concepts(concepts: Dict[str, dict]) -> List[dict]:
    results = []
    for concept_name, item in sorted(concepts.items()):
        labels = {
            role: dict(sorted(lang_map.items()))
            for role, lang_map in sorted(item["labels"].items())
        }
        code = labels.get(TERSE_ROLE, {}).get("en")
        zh_tw = labels.get(LABEL_ROLE, {}).get("zh-tw")
        en = labels.get(LABEL_ROLE, {}).get("en")
        total_zh = labels.get(TOTAL_ROLE, {}).get("zh-tw")
        total_en = labels.get(TOTAL_ROLE, {}).get("en")

        presentation = dedupe_records(
            item["presentation"],
            ["role", "parent_concept", "order", "preferred_label", "source_file", "path"],
        )
        calculation_children = dedupe_records(
            item["calculation_children"],
            ["role", "concept_name", "order", "weight", "source_file"],
        )
        calculation_parents = dedupe_records(
            item["calculation_parents"],
            ["role", "concept_name", "order", "weight", "source_file"],
        )
        definition_children = dedupe_records(
            item["definition_children"],
            ["role", "arcrole", "concept_name", "order", "source_file"],
        )
        definition_parents = dedupe_records(
            item["definition_parents"],
            ["role", "arcrole", "concept_name", "order", "source_file"],
        )

        search_parts = [
            concept_name,
            item["name"],
            zh_tw,
            en,
            code,
            total_zh,
            total_en,
            item["period_type"],
            item["balance"],
            item["type"],
            item["substitution_group"],
            " ".join(sorted(item["families"])),
            " ".join(sorted(filter(None, item["role_occurrences"]))),
        ]
        for entry in presentation:
            if "path" in entry:
                search_parts.append(" > ".join(entry["path"]))

        results.append(
            {
                "concept_name": concept_name,
                "namespace_prefix": item["namespace_prefix"],
                "name": item["name"],
                "id": item["id"],
                "type": item["type"],
                "substitution_group": item["substitution_group"],
                "abstract": item["abstract"],
                "nillable": item["nillable"],
                "period_type": item["period_type"],
                "balance": item["balance"],
                "source_kind": item["source_kind"],
                "source_hrefs": sorted(item["source_hrefs"]),
                "source_files": sorted(item["source_files"]),
                "families": sorted(item["families"]),
                "roles": sorted(filter(None, item["role_occurrences"])),
                "zh_tw": zh_tw,
                "en": en,
                "code": code,
                "labels": labels,
                "presentation": presentation,
                "calculation_children": calculation_children,
                "calculation_parents": calculation_parents,
                "definition_children": definition_children,
                "definition_parents": definition_parents,
                "search_text": " | ".join(dict.fromkeys(part for part in search_parts if part)),
            }
        )
    return results


def build_compact_index(items: List[dict]) -> List[dict]:
    compact = []
    for item in items:
        compact.append(
            {
                "concept_name": item["concept_name"],
                "en": item["en"],
                "zh_tw": item["zh_tw"],
                "code": item["code"],
                "families": item["families"],
                "roles": item["roles"],
            }
        )
    return compact


def build_summary(root_dir: Path, categories: Dict[str, List[Path]], items: List[dict]) -> dict:
    family_counts = Counter()
    for item in items:
        for family in item["families"]:
            family_counts[family] += 1
    return {
        "root_dir": str(root_dir),
        "file_counts": {name: len(paths) for name, paths in categories.items()},
        "concept_count": len(items),
        "family_concept_counts": dict(sorted(family_counts.items())),
    }


def run_directory_mode(root_dir: Path, output: Path, compact_output: Path, summary_output: Path) -> None:
    categories = discover_files(root_dir)
    concepts: Dict[str, dict] = {}

    for path in categories["xsd"]:
        parse_xsd(path, root_dir, concepts)
    for path in categories["label"]:
        parse_label(path, root_dir, concepts)
    for path in categories["presentation"]:
        parse_presentation(path, root_dir, concepts)
    for path in categories["calculation"]:
        parse_calculation(path, root_dir, concepts)
    for path in categories["definition"]:
        parse_definition(path, root_dir, concepts)

    full_items = finalize_concepts(concepts)
    compact_items = build_compact_index(full_items)
    summary = build_summary(root_dir, categories, full_items)

    output.parent.mkdir(parents=True, exist_ok=True)
    compact_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(full_items, ensure_ascii=False, indent=2), encoding="utf-8")
    compact_output.write_text(json.dumps(compact_items, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Root parsed: {root_dir}")
    print(f"Full dictionary written to: {output}")
    print(f"Compact dictionary written to: {compact_output}")
    print(f"Summary written to: {summary_output}")
    print(f"File counts: {summary['file_counts']}")
    print(f"Concept count: {summary['concept_count']}")


def run_single_set_mode(
    xsd: Path,
    label: Path,
    presentation: Path,
    calculation: Path,
    output: Path,
    compact_output: Path,
    summary_output: Path,
) -> None:
    root_dir = xsd.parent.resolve()
    concepts: Dict[str, dict] = {}
    parse_xsd(xsd, root_dir, concepts)
    parse_label(label, root_dir, concepts)
    parse_presentation(presentation, root_dir, concepts)
    parse_calculation(calculation, root_dir, concepts)

    full_items = finalize_concepts(concepts)
    compact_items = build_compact_index(full_items)
    summary = build_summary(
        root_dir,
        {
            "xsd": [xsd],
            "label": [label],
            "presentation": [presentation],
            "calculation": [calculation],
            "definition": [],
        },
        full_items,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    compact_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(full_items, ensure_ascii=False, indent=2), encoding="utf-8")
    compact_output.write_text(json.dumps(compact_items, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Full dictionary written to: {output}")
    print(f"Compact dictionary written to: {compact_output}")
    print(f"Summary written to: {summary_output}")
    print(f"Concept count: {summary['concept_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse XBRL taxonomy files into JSON dictionaries.")
    parser.add_argument("--root-dir", help="Root taxonomy directory. When set, all supported files below it are parsed.")
    parser.add_argument("--xsd", help="Path to a taxonomy XSD file.")
    parser.add_argument("--label", help="Path to a label XML file.")
    parser.add_argument("--presentation", help="Path to a presentation XML file.")
    parser.add_argument("--calculation", help="Path to a calculation XML file.")
    parser.add_argument(
        "--output",
        default="src/services/xbrl_data_dictionary.json",
        help="Output JSON path for the full dictionary.",
    )
    parser.add_argument(
        "--compact-output",
        default="src/services/xbrl_account_title_compact.json",
        help="Output JSON path for the compact concept/label/code mapping.",
    )
    parser.add_argument(
        "--summary-output",
        default="src/services/xbrl_dictionary_summary.json",
        help="Output JSON path for parsing summary.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    compact_output = Path(args.compact_output)
    summary_output = Path(args.summary_output)

    if args.root_dir:
        run_directory_mode(Path(args.root_dir), output, compact_output, summary_output)
        return

    missing = [name for name in ["xsd", "label", "presentation", "calculation"] if not getattr(args, name)]
    if missing:
        parser.error("Either --root-dir or all of --xsd/--label/--presentation/--calculation are required.")

    run_single_set_mode(
        Path(args.xsd),
        Path(args.label),
        Path(args.presentation),
        Path(args.calculation),
        output,
        compact_output,
        summary_output,
    )


if __name__ == "__main__":
    main()
