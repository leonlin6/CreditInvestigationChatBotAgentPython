# CreditInvestigationChatBotPython

This project is a backend server for a Credit Investigation ChatBot, implemented in Python 3.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python app.py
   ```

## Project Structure
- `app.py`: Main entry point for the backend server.
- `requirements.txt`: Python dependencies.

## XBRL To SQL

Use `scripts/build_xbrl_sql.py` to parse a `tifrs-20200630` taxonomy directory together with one XBRL instance file, then generate SQL `INSERT` statements for these tables:

- `report_instance`
- `taxonomy_entry_point`
- `taxonomy_concept`
- `taxonomy_presentation`
- `taxonomy_calculation`
- `xbrl_fact`
- `field_dictionary`
- `field_concept_mapping`
- `financial_metric_value`

Example:

```bash
python3 scripts/build_xbrl_sql.py \
  --taxonomy-root /path/to/tifrs-20200630 \
  --instance /path/to/report.xbrl \
  --sql-output ./output/report_import.sql
```

If you only want taxonomy SQL first, `--instance` is optional:

```bash
python3 scripts/build_xbrl_sql.py \
  --taxonomy-root /path/to/tifrs-20200630 \
  --sql-output ./output/taxonomy_only.sql
```

By default, `field_dictionary` and `field_concept_mapping` are created in `auto-concept` mode, which means each parsed concept is mapped 1:1 to an internal field. If you already have your own business field definitions, switch to `--field-mode custom` and provide a JSON file shaped like:

```json
{
  "fields": [
    {
      "field_id": "assets",
      "canonical_name": "assets",
      "zh_name": "資產總額",
      "en_name": "Assets",
      "module": "BSCI",
      "statement_type": "balance_sheet",
      "value_type": "numeric",
      "description": "Total assets"
    }
  ],
  "mappings": [
    {
      "field_id": "assets",
      "taxonomy_id": "tifrs-20200630:BSCI",
      "concept_ids": ["ifrs-full_Assets"],
      "industry_type": "CI",
      "priority": 1,
      "effective_from": "2020-06-30",
      "effective_to": null
    }
  ]
}
```
