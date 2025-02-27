import torch
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(message)s')
logger = logging.getLogger(__name__)


def convert_tubes(tube, tubekeys=('b', 'm', 't')):
    """
    For each tube type, filter out extremely high values, then apply arcsinh transform to the remaining values.
    """
    tube_dat = torch.load(tube, weights_only=False)
    for tube_type in tubekeys:
        tube = tube_dat[tube_type]

        # Remove extreme values
        temp = tube[(tube < 2**20 - 1000).all(axis=1)]
    
        # Compensate after removing extreme values but before arcsinh transform
        spillover = torch.tensor(tube_dat[f'{tube_type}_spillover'])
        compmat = torch.linalg.inv(spillover).T
        # bfloat not supported on some platforms (like cpu), so do this in high-precision and then convert back later
        temp = temp.float() @ compmat.float() 

        # first 2 columns must be FS-H & FS-A & last col is Time.
        # exclude fwd scatt & time columns from arcsinh transf
        temp[:, 2:-1] = torch.arcsinh(temp[:, 2:-1] / 300)
        # rescale FS to more closely match arcsinh transform
        temp[:, :2] = temp[:, :2] * 9 / 1.894880e5
        # drop time col & overwrite input tensor

        # We don't need a huge range here, but more precision might be nice, so use fp16 instead of bfloat16
        tube_dat[tube_type] = temp[:, :-1].half() 

    return tube_dat

def process_tube(tube, destdir="."):
    torch.set_num_threads(4)
    try:
        logger.info(f"Converting {tube}")
        result = convert_tubes(tube)
        parentdir = os.path.dirname(tube)
        newname = os.path.basename(tube).replace("_raw", "").replace('.pt', '_converted.pt')
        torch.save(result, os.path.join(destdir, newname))
        logger.info(f"Saved {newname}")
    except Exception as e:
        logger.error(f"Error converting {tube}: {e}")
        raise e

def main(tubedir, workers=4):
    # process_tube("/Users/brendan.ofallon/data/flow/muir_raw_tubedata/19217129080_raw.pt")
    paths = glob.glob(os.path.join(tubedir, '*.pt'))
    logger.info(f"Found {len(paths)} files to convert")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(process_tube, paths, chunksize=512)


if __name__ == "__main__":
    main(sys.argv[1])

