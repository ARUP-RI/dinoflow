import pandas as pd
import sys
import csv


SYSTEM_PROMPT="""You are a helpful and intelligent board-certified hematopathologist with decades of clinical experience."""

PROMPT="""
The above report is from a flow cytometry analysis from a leukemia / lymphoma panel. Please determine if the report indicates that the patient likely
has acute myeloid leukemia by assessing the degree of atypical myeloid blasts. If there is no evidence of myeloid leukemia, answer 'Low'. 
If there are low levels of myeloid blasts, less than 20% of viable leukocytes but more than normal levels, answer 'Medium'. 
If greater than 20% of viable leukocytes are myeloid blasts in a manner consistent with acute myeloid leukemia, answer 'High'. 
Answer only with 'Low', 'Medium' or 'High'.   
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
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, escapechar='\\', quoting=csv.QUOTE_ALL)
        writer.writerow(['ACCESSION', 'CHART_COMMENT_TEXT', 'prediction', 'label', 'Reasoning', 'Result'])

        for index, row in df.iterrows():
            report = row['CHART_COMMENT_TEXT']
            sys.stderr.write(f"Processing {index}: {row['ACCESSION']}\n")
            response = run_ollama(report)
            reasoning, result = response['message']['content'].split("</think>")
            writer.writerow([row['ACCESSION'], row['CHART_COMMENT_TEXT'], row['prediction'], row['AML'], reasoning, result])


if __name__ == '__main__':
    main(sys.argv[1])
