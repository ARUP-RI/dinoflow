import pandas as pd
import sys
import csv

SYSTEM_PROMPT = """You are a careful, literal hematopathology assistant. When asked to extract values from a report, you return only the requested fields in strict JSON and nothing else."""

#SYSTEM_PROMPT="""You are a helpful and intelligent board-certified hematopathologist with decades of clinical experience."""

# PROMPT="""
# The above report is from a flow cytometry analysis. Please extract from the report the disease burden of the patient.   
# """

PROMPT = r"""
You will extract two numeric percentages from the flow cytometry report that appears ABOVE this message (that is the **case report**). Any text below is guidance and synthetic examples—DO NOT extract from examples.

<Task>
Return:
- disease_burden_pct  → % of the abnormal/monoclonal/atypical population in THIS CASE.
- viability_pct       → sample viability % reported for THIS CASE.
</Task>

<Definitions & rules>
- Prefer the IMPRESSION for disease burden. Common phrasings:
  "accounting for X% of viable leukocytes", "comprise X% of viable leukocytes",
  "represent X% of the total leukocytes", "constitute X%".
- If the IMPRESSION lacks a percentage, fall back to POPULATION PHENOTYPE or ANALYSIS
  lines that explicitly quantify the abnormal population. Ignore generic differentials
  (e.g., "Lymphocytes 70%") unless it is explicitly the abnormal clone.
- If the report states that **no abnormal population is identified**, set disease_burden_pct = 0.
- Viability may appear as "Viability: 96%", "Viability 96%", "Viability96:%", or "<0.1%".
  If a number uses "<", drop the symbol and keep the numeric value (e.g., "<0.1%" → 0.1).
- Ignore historical comparisons and other accessions/dates in COMMENT
  (e.g., "were last reported", "similar study", prior percentages).
- If a field is not present, use null for that field.
- Percentages must be numeric 0–100 (no "%" sign in the JSON).

<Output format>
- Return ONLY this JSON, wrapped in <final>…</final> (no extra text):
{
  "disease_burden_pct": number | 0 | null,
  "viability_pct": number | null
}

<Examples>  <!-- Learn pattern only; do not extract from these -->

<ExampleReport A>
SAMPLE: PERIPHERAL BLOOD

IMPRESSION:
CD5 positive B-cell lymphoproliferative disorder (comprise 55% of viable leukocytes)...
...
Viability96:%
</ExampleReport A>
<ExpectedJSON A>
<final>{
  "disease_burden_pct": 55,
  "viability_pct": 96
}</final>
</ExpectedJSON A>

<ExampleReport B>
SAMPLE: PERIPHERAL BLOOD

IMPRESSION:
Lambda restricted partial CD10+ B cell population accounting for 25% of the viable leukocytes...
...
Viability: 96%
</ExampleReport B>
<ExpectedJSON B>
<final>{
  "disease_burden_pct": 25,
  "viability_pct": 96
}</final>
</ExpectedJSON B>

<ExampleReport C>
SAMPLE: PERIPHERAL BLOOD

IMPRESSION:
1. Relative increase of granulocytes without immunophenotypic aberrancy.
2. No abnormal B cell, T cell, NK cell, or plasma cell population identified.
...
Viability: 92%
</ExampleReport C>
<ExpectedJSON C>
<final>{
  "disease_burden_pct": 0,
  "viability_pct": 92
}</final>
</ExpectedJSON C>

<ExampleReport D>
SAMPLE: BONE MARROW

IMPRESSION:
Atypical B lymphoblasts accounting for 40% of viable leukocytes. See comment.
COMMENT:
Phenotypically similar cells were last reported on 6-10-2023 where they accounted for 0.16%...
</ExampleReport D>
<ExpectedJSON D>
<final>{
  "disease_burden_pct": 40,
  "viability_pct": null
}</final>
</ExpectedJSON D>
"""


from ollama import Client
client = Client(host='http://localhost:11434')

def run_ollama(report):
    response = client.chat(model='deepseek-r1:70b', messages=[
        {'role': 'system',
            'content': SYSTEM_PROMPT,
        },
        {'role': 'user',
            'content': "Here is a flow cytometry report: " + report + "\n\n" + PROMPT,
        },
    ])
    return response

def main(path):
    df = pd.read_csv(path)
    # Use a CSV writer to write the results to a file
    with open('/home/32210/test_things/lmd_conversion/results_ollama_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, escapechar='\\', quoting=csv.QUOTE_ALL)
        writer.writerow(['ACCESSION', 'chart_text', 'label', 'Reasoning', 'Result'])

        for index, row in df.iterrows():
            report = row['chart_text']
            sys.stderr.write(f"Processing {index}: {row['ACCESSION']}\n")
            response = run_ollama(report)
            reasoning, result = response['message']['content'].split("</think>")
            writer.writerow([row['ACCESSION'], row['chart_text'], row['ACTION_REQUIRED'], reasoning, result])


if __name__ == '__main__':
    main(sys.argv[1])
