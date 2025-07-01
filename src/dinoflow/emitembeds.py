import os
import logging
import torch
import typer
from torch.utils.data import DataLoader

from dinoflow.models import load_checkpoint
from dinoflow.data import TubeData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer(pretty_exceptions_show_locals=False)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]   %(levelname)s   %(message)s')
logger = logging.getLogger(__name__)

@app.command()
def compute_embeddings(backbone: str,
                      input_csv: str,
                      output_dir: str,
                      tube_type: str,
                      dataroot: str = ".",
                      events: int = 4096,
                      batch_size: int = 16):
    """
    Compute embeddings for each sample in the input CSV using a backbone model.
    The embeddings will be saved as PyTorch tensors in the output directory.
    
    Args:
        backbone: Path to the backbone model checkpoint
        input_csv: Path to the input CSV file containing sample information
        output_dir: Directory to save the embeddings
        tube_type: Type of tube to process ('b', 't', or 'm')
        dataroot: Root directory for the data
        events: Number of events to use per sample
        batch_size: Batch size for processing
    """
    assert tube_type in ['b', 't', 'm'], f"Invalid tube type: {tube_type}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load backbone model
    logger.info(f"Loading backbone from {backbone}")
    backbone_model, _ = load_checkpoint(backbone)
    backbone_model.eval().to(DEVICE)
    
    # Setup dataset and dataloader
    dataset = TubeData(input_csv, 
                      data_root=dataroot,
                      labelkey="path", # This is ignored but we must provide a value
                      tubes_to_return=[tube_type],
                      events_to_return=int(events))
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4)
    logger.info(f"Loaded {len(dataset)} samples for embedding computation")
    
    # Process each batch
    with torch.inference_mode():
        for batch_idx, (batch, rowdict) in enumerate(dataloader):
            # Move batch to device
            batch = batch.float().to(DEVICE)
            
            embeddings = backbone_model(batch)
            
            for i in range(len(batch)):
                sample_idx = batch_idx * batch_size + i
                accession = rowdict['accession'][i]
                
                output = {
                    'embeddings': embeddings[i].cpu(),
                    'tube_type': tube_type
                }
                
                output_path = os.path.join(output_dir, f"{accession}_{tube_type}_embeds.pt")
                torch.save(output, output_path)
                
                if sample_idx % 100 == 0:
                    logger.info(f"Processed {sample_idx} samples")

if __name__ == "__main__":
    app() 
