#!/usr/bin/env python3
import sys, json, csv, re, time
import pandas as pd
from ollama import Client


# Config
MODEL = "gemma3:27b"
HOST = "http://localhost:11434"
MAX_TRIES = 3

INPUT_TEXT_COL = "chart_text"
INPUT_ID_COL = "ACCESSION"

OUT_JSONL = "/home/31792/flow3_testset.jsonl"
OUT_CSV  = "/home/31792/flow3_testset.csv"

SYSTEM_PROMPT = """
You are extracting structured labels from a flow cytometry report excerpt.

GOAL:
- Infer labels ONLY from statements about the CURRENT specimen’s flow cytometry findings.
- Do NOT use patient history, prior tests, prior flow results, outside reports, or speculative discussion as evidence unless explicitly tied to the current specimen.

SCOPE / TEMPORAL FILTER (IMPORTANT):
- Base all labels ONLY on findings describing the CURRENT specimen.
- IGNORE any discussion of prior tests, previous flow cytometry, historical comparisons, outside reports, or follow-up recommendations.
- If abnormal findings are mentioned ONLY in historical context and not attributed to the current specimen, treat them as NOT PRESENT.
- If the report states "no abnormal population identified" for the current specimen, abnormal_population must be 0 even if prior abnormalities are discussed.
- If it is unclear whether a finding applies to the current specimen, assume it does NOT.

HIERARCHY RULES:
1) If abnormal_population = 0:
   malignant_vs_reactive must be "none"
   lineage must be "none"
   clonality must be "none"
   maturation must be "none"
   aberrancy_grade must be 0

2) clonality applies only if lineage is "B" or "T". Otherwise clonality must be "none".

3) maturation applies only if abnormal_population = 1. Otherwise maturation must be "none".

NOTE ON "normal" VS "quality" (IMPORTANT):
- normal reflects biology/interpretation (no abnormal population and no true indeterminate language).
- suboptimal_viability reflects specimen quality only and does NOT change normal by itself.
- A case may have normal=1 and suboptimal_viability=1 simultaneously.

DATA QUALITY:
- Extract viability percentage ONLY if explicitly present in the excerpt.
- viability_pct: number 0–100, or null if not present.
- suboptimal_viability: 1 if viability_pct < 80 else 0.
- If viability not present, set viability_pct=null and suboptimal_viability=0.

IMPORTANT SIZE/WORDING NOTE:
- Do NOT down-grade abnormal_population, clonality, malignant_vs_reactive, or aberrancy_grade based solely on population size, low percentage, or wording like "minute", "low-level", or "small".
- If explicit clonality/restriction is present, treat it as a definitive abnormal finding for the CURRENT specimen unless the report explicitly states it is reactive or polyclonal.

FLOW INDETERMINATE:
- flow_indeterminate captures true uncertainty/limitations in the CURRENT specimen that make a flow-negative result not fully reassuring.
- Set flow_indeterminate=1 if the excerpt contains any of:
  - "cannot exclude"
  - "indeterminate"
  - "equivocal"
  - "limited evaluation" / "limited by" / "suboptimal specimen"
  - "not specific and may be observed in reactive settings as well as ... neoplasms/myeloid neoplasms"
  - Explicit recommendation for repeat testing due to limitations
  - "no definitive immunophenotypic aberrancy" ONLY when accompanied by other uncertainty/limitation language
- Set flow_indeterminate=0 otherwise.
- IMPORTANT: Boilerplate correlation statements alone do NOT set flow_indeterminate=1:
  - "Clinical and morphologic correlation will be required"
  - "Flow cytometry is an ancillary study"
  - "Clinical correlation is recommended"
  - Generic correlation language WITHOUT "cannot exclude/indeterminate/equivocal/limited"

NORMAL LABELING:
- normal reflects whether the CURRENT specimen is interpreted as normal/negative by flow findings.
- Set normal=1 only when:
  - abnormal_population=0 AND flow_indeterminate=0
- Set normal=0 when:
  - abnormal_population=1 OR flow_indeterminate=1
- suboptimal_viability does NOT change normal by itself.

LABEL DEFINITIONS / RUBRICS

A) abnormal_population (0/1)
Set abnormal_population=1 if the CURRENT specimen explicitly indicates an abnormal / atypical / clonal / phenotypically abnormal population is present.
Common evidence phrases:
- "abnormal population identified"
- "phenotypically abnormal [B/T/myeloid/blast] population"
- "atypical population"
- "monoclonal population"
- "light chain restriction"
- "kappa restricted" / "lambda restricted"
- "restricted TRBC/TRBC1 expression"
- "CLL-like phenotype" / "CLL/SLL-like phenotype"
- "MBL-like population"
- "increased blasts with aberrant phenotype"
- "aberrant blasts" / "aberrant myeloid blasts"
- "abnormal myeloid blasts"

Set abnormal_population=0 if the CURRENT specimen explicitly indicates none, or only reactive/physiologic patterns without calling any population abnormal.
Common evidence phrases:
- "no abnormal ... population identified"
- "no immunophenotypic aberrancy"
- "no evidence of lymphoma/leukemia by flow cytometry"
- "reactive changes only" WITHOUT an abnormal population call
- Left shift / granulocytosis / monocytosis WITHOUT immunophenotypic aberrancy does NOT by itself make abnormal_population=1.

B) malignant_vs_reactive ("malignant" | "reactive" | "none")
Only assign malignant/reactive when abnormal_population=1 (otherwise "none").

Set to "malignant" if the CURRENT specimen describes a neoplastic/clonal process or uses definitive malignant framing.
Evidence phrases:
- "consistent with lymphoma/leukemia"
- "diagnostic of"
- "neoplastic"
- "clonal B-cell population"
- "light chain restricted B-cells" / "kappa restricted" / "lambda restricted"
- "aberrant T-cell population with restricted TRBC/TRBC1"
- "blasts with aberrant phenotype consistent with acute leukemia"
- "plasma cell neoplasm" / "monoclonal plasma cells" (if explicitly stated by flow)
- "CLL-like phenotype" / "CLL/SLL-like phenotype" (when tied to a restricted/clonal population)

Set to "reactive" if the CURRENT specimen describes an abnormal/atypical population but explicitly frames it as reactive/benign/inflammatory OR explicitly says it may be reactive and not diagnostic of malignancy.
Evidence phrases:
- "reactive"
- "likely reactive"
- "favored reactive"
- "benign"
- "no definitive evidence of lymphoproliferative disorder"
- "atypical ... of uncertain significance" / "uncertain significance" (ONLY if clonality/restriction is NOT explicitly present)

DECISION RULES (IMPORTANT):
- If clonality="clonal" (e.g., kappa/lambda restriction or restricted TRBC/TRBC1), then malignant_vs_reactive MUST be "malignant"
  EVEN IF language like "minute", "low-level", "small", or "of uncertain significance" is present,
  UNLESS the report explicitly states the population is reactive or polyclonal.
- If reactive/uncertain language is present AND there is NO explicit clonality/restriction, set malignant_vs_reactive="reactive".
- If neither malignant nor reactive framing is present, choose "reactive" (conservative).

C) lineage ("B" | "T" | "myeloid" | "none")
Only assign lineage when abnormal_population=1 (otherwise "none").

Choose based on which abnormal population is described:
- "B" if abnormal B-cells / light chain restriction / monoclonal B-cell population is described.
  Clues: kappa/lambda restriction, "B-cell population abnormal", "CLL-like phenotype"
- "T" if abnormal T-cells / TRBC/TRBC1 restriction / aberrant T phenotype is described.
  Clues: CD3/CD2/CD5/CD7/CD4/CD8 abnormalities, TRBC restriction
- "myeloid" if abnormal myeloid/blast population is described.
  Clues: "myeloid blasts", "abnormal myeloid population", "aberrant blasts", aberrant CD34/CD117/HLA-DR/myeloid markers described as abnormal
- If multiple abnormal lineages are explicitly present, choose the one emphasized as the primary abnormal population; if truly co-dominant, prefer "myeloid" only when blasts/acute leukemia is explicitly described; otherwise prefer the clearly clonal lymphoid lineage.

D) clonality ("clonal" | "polyclonal" | "uncertain" | "none")
Rule: clonality applies only if lineage is "B" or "T". If lineage is not "B" or "T", clonality="none".

IMPORTANT GUARDRAILS:
- Do NOT infer clonality from kappa:lambda or CD4:CD8 ratios alone unless the report explicitly uses restriction/monotypic/clonal language.
- If the excerpt explicitly states "no abnormal B cell population identified" (or equivalent), do NOT set B-cell clonality from ratio skew alone.

For lineage="B":
- "clonal" if clear light-chain restriction or explicit monoclonality.
  Evidence: "light chain restricted", "kappa restricted", "lambda restricted", "monoclonal B-cells", "monotypic"
- "polyclonal" if explicitly states polytypic/polyclonal light chains.
  Evidence: "polytypic", "polyclonal", "no light chain restriction", "polytypic kappa and lambda"
- "uncertain" if equivocal skew, limited sample, or hedged language WITHOUT definitive restriction.
  Evidence: "suggests", "cannot exclude restriction", "equivocal", "limited evaluation"

For lineage="T":
- "clonal" if restricted TRBC/TRBC1 or explicit T-cell clonality language.
  Evidence: "restricted TRBC/TRBC1", "restricted TRBC", "clonal T-cell population"
- "polyclonal" if explicitly states no evidence of clonality and/or broad TRBC expression consistent with polyclonal.
- "uncertain" if described as atypical/TCUS/uncertain significance without definitive restriction.

E) maturation ("acute" | "mature" | "none")
Rule: maturation applies only if abnormal_population=1. If abnormal_population=0, maturation="none".

Set maturation="acute" if the CURRENT specimen explicitly describes an abnormal blast population or acute leukemia features.
Evidence phrases:
- "increased blasts"
- "myeloid blasts"
- "blasts with aberrant phenotype"
- "aberrant blasts"
- "CD34+ blasts" / "CD117+ blasts" (when described as an abnormal/blast population)
- "consistent with acute leukemia"
- Explicit "acute myeloid leukemia" / "AML" / "B-ALL" / "T-ALL" in the impression tied to flow findings

IMPORTANT GUARDRAIL:
- Do NOT set maturation="acute" solely because the differential lists a small baseline blast percentage (e.g., "Blasts: 0.03%") unless blasts are described as increased/abnormal/aberrant.

Set maturation="mature" if the CURRENT specimen describes an abnormal population that is NOT blasts/acute.
Examples:
- "CLL/SLL-like phenotype"
- "monoclonal B-cell population"
- "aberrant T-cell population" (without blasts wording)
- "plasma cell neoplasm" / monoclonal plasma cells
- "hairy cell leukemia"
- "LGL"

If abnormal_population=1 but the excerpt does not make it clear whether blasts vs mature, default maturation="mature" unless any acute/blast evidence is present.

F) aberrancy_grade (0 | 1 | 2)
Grade 0 (none):
  - Explicitly normal / no abnormal population identified, OR only physiologic/reactive patterns without aberrant phenotype.

Grade 1 (mild / subtle / atypical):
  - Small or equivocal abnormal population OR mild antigen shifts without a classic aberrant pattern.
  - NOTE: If explicit clonality/restriction is present, do NOT use Grade 1 solely because the population is small.

Grade 2 (clear / classic aberrancy):
  - Definite abnormal population with classic aberrancy OR clear clonality OR definitive malignant framing.
  - Any explicit light-chain restriction (kappa or lambda) or restricted TRBC/TRBC1 automatically qualifies as aberrancy_grade=2.

TIE-BREAKER:
- If abnormal_population=1 but language is equivocal/uncertain and there is NO explicit clonality/restriction, default aberrancy_grade=1.
- If explicit clonality/restriction is present, aberrancy_grade=2.

OUTPUT:
Return ONLY valid JSON with EXACT keys and allowed values:
{
  "abnormal_population": 0 or 1,
  "flow_indeterminate": 0 or 1,
  "normal": 0 or 1,
  "malignant_vs_reactive": "malignant" | "reactive" | "none",
  "lineage": "B" | "T" | "myeloid" | "none",
  "clonality": "clonal" | "polyclonal" | "uncertain" | "none",
  "maturation": "acute" | "mature" | "none",
  "aberrancy_grade": 0 | 1 | 2,
  "viability_pct": number or null,
  "suboptimal_viability": 0 or 1
}

IMPORTANT:
- Compute flow_indeterminate based ONLY on the rules above (not on boilerplate).
- Compute "normal" AFTER determining abnormal_population and flow_indeterminate:
  - If abnormal_population == 1 → normal = 0
  - Else if flow_indeterminate == 1 → normal = 0
  - Else → normal = 1
- suboptimal_viability does NOT change normal by itself.

No extra text. No markdown.
"""


# Helpers functions
def extract_json_object(text: str):
    """Find first '{' and last '}' and parse JSON. Returns dict or raises."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("No JSON object found in output")
    chunk = text[s:e+1]
    return json.loads(chunk)


def extract_viability_regex(report_text: str):
    """Safety net: pull viability % from report text if present."""
    patterns = [
        r"\bviability\b[^0-9]{0,20}(\d{1,3}(?:\.\d+)?)\s*%",
        r"\bcell viability\b[^0-9]{0,20}(\d{1,3}(?:\.\d+)?)\s*%",
    ]
    t = report_text.lower()
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return v
    return None


def coerce_int01(x, default=0):
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, (int, float)):
        return 1 if int(x) != 0 else 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y"}: return 1
        if s in {"0", "false", "no", "n", ""}: return 0
    return default


def normalize_enum(val, allowed, default):
    if val is None: return default
    if isinstance(val, str):
        s = val.strip().lower()
        aliases = {
            "malignancy": "malignant",
            "neoplastic": "malignant",
            "benign": "none",
            "negative": "none",
            "normal": "none",
            "n/a": "none",
            "na": "none",
            "equivocal": "uncertain",
        }
        s = aliases.get(s, s)
        for a in allowed:
            if s == a.lower():
                return a
    return default


def apply_hierarchy(labels: dict, report_text: str):
    out = dict(labels)

    # abnormal population is now the ONLY gate
    out["abnormal_population"] = coerce_int01(out.get("abnormal_population", 0), 0)

    out["malignant_vs_reactive"] = normalize_enum(
        out.get("malignant_vs_reactive"),
        ["malignant", "reactive", "none"],
        "none",
    )
    out["lineage"] = normalize_enum(out.get("lineage"), ["B", "T", "myeloid", "none"], "none")
    out["clonality"] = normalize_enum(
        out.get("clonality"),
        ["clonal", "polyclonal", "uncertain", "none"],
        "none",
    )

    # aberrancy grade
    ag = out.get("aberrancy_grade", 0)
    try:
        ag = int(float(ag))
    except Exception:
        ag = 0
    if ag not in (0, 1, 2):
        ag = 0
    out["aberrancy_grade"] = ag

    # viability
    v = out.get("viability_pct", None)
    v_val = None
    if isinstance(v, (int, float)):
        v_val = float(v)
    elif isinstance(v, str):
        try:
            v_val = float(v.strip().replace("%", ""))
        except Exception:
            pass

    if v_val is None:
        v_val = extract_viability_regex(report_text)

    if v_val is not None:
        v_val = max(0.0, min(100.0, v_val))
    out["viability_pct"] = v_val
    out["suboptimal_viability"] = 1 if (v_val is not None and v_val < 80.0) else 0

    # hierarchy 
    if out["abnormal_population"] == 0:
        out.update({
            "malignant_vs_reactive": "none",
            "lineage": "none",
            "clonality": "none",
            "aberrancy_grade": 0,
        })

    if out["lineage"] not in ("B", "T"):
        out["clonality"] = "none"

    return out

def run_ollama_chat(client: Client, report_text: str):
    return client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report_text},
        ],
    )


def extract_with_retries(client: Client, report_text: str, max_tries: int = 3):
    last_raw = ""
    for attempt in range(1, max_tries + 1):
        resp = run_ollama_chat(client, report_text)
        last_raw = resp["message"]["content"]

        try:
            parsed = extract_json_object(last_raw)
            return parsed, last_raw
        except Exception:
            # Repair: ask it to output JSON only from its previous output
            repair_system = "Return ONLY valid JSON. No extra text."
            repair_user = f"Convert this to valid JSON with the required keys only:\n\n{last_raw}"
            resp2 = client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": repair_user},
                ],
            )
            last_raw = resp2["message"]["content"]
            try:
                parsed = extract_json_object(last_raw)
                return parsed, last_raw
            except Exception:
                time.sleep(0.5 * attempt)

    return None, last_raw


def main(path: str):
    df = pd.read_csv(path)
    client = Client(host=HOST)

    # JSONL output (recommended)
    out_jsonl_f = open(OUT_JSONL, "w", encoding="utf-8")
    # Optional CSV output (nice for quick eyeballing)
    out_csv_f = open(OUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        out_csv_f,
        fieldnames=[
            "ACCESSION",
            "ok",
            "action_required",
            "abnormal_population",
            "malignant_vs_reactive",
            "lineage",
            "clonality",
            "aberrancy_grade",
            "viability_pct",
            "suboptimal_viability",
            "flow_indeterminate",
            "normal",
            "maturation",
        ],
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )
    writer.writeheader()

    ok, fail = 0, 0
    for i, row in df.iterrows():
        rid = str(row[INPUT_ID_COL])
        report_text = str(row[INPUT_TEXT_COL])

        sys.stderr.write(f"Processing {i}: {rid}\n")

        parsed, raw = extract_with_retries(client, report_text, max_tries=MAX_TRIES)
        if parsed is None:
            fail += 1
            rec = {"ACCESSION": rid, "ok": False, "raw_output": raw}
            out_jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        labels = apply_hierarchy(parsed, report_text=report_text)
        labels_out = {"ACCESSION": rid, "ok": True, **labels}

        out_jsonl_f.write(json.dumps(labels_out, ensure_ascii=False) + "\n")
        writer.writerow(labels_out)
        ok += 1

    out_jsonl_f.close()
    out_csv_f.close()
    sys.stderr.write(f"Done. ok={ok} fail={fail}\n")
    sys.stderr.write(f"Wrote: {OUT_JSONL}\nWrote: {OUT_CSV}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_mtl_labels.py <reports.csv>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
