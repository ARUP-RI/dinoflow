import csv
import sys
from pathlib import Path

def main(csvpath, datadir):
    """
    Find the data in the datadir that matches the csvpath
    """
    tubedata = Path(datadir).glob('*.pt')
    tubemap = {}
    for tube in tubedata:
        acc = tube.stem.split('_')[0]
        tubemap[acc] = tube

    with open(csvpath, 'r') as f:
        reader = csv.DictReader(f)
        print("accession,path,normal,label")
        for row in reader:
            acc = row['accession']
            if acc in tubemap:
                print(f"{acc},{tubemap[acc]},{row['NORMAL']},{row['AML']}")
            else:
                print(f"{acc} NOT FOUND")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
