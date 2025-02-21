import csv
import torch
import sys

def main(label_file, min_events, tube='m'):
    with open(label_file, 'r') as f:
        reader = csv.DictReader(f)
        print(reader.fieldnames)
        for row in reader:
            tubedata = torch.load(row['path'], weights_only=True)
            if tubedata[tube].shape[0] >= min_events:
                print(",".join(row[k] for k in reader.fieldnames))
            else:
                sys.stderr.write(f"Skipping {row['path']} because it has less than {min_events} events\n")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
