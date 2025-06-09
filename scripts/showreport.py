
import sys
import pandas as pd

from rich import print

acc = sys.argv[1]

casedx_file = "/data2/brendan/flow/casedx_2024-08-21_chart_text.csv"
dx = pd.read_csv(casedx_file, dtype={"ACCESSION": str})

r = dx[dx ['ACCESSION'] == acc ]
if len(r) == 0:
    print("No report found")
else:
    r = r.iloc[0]

print(r['CHART_COMMENT_TEXT'])
