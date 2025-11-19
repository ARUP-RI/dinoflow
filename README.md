
## DinoFlow - DINO algorithm for Flow Cytometry data

DinoFlow is an algorithm for generation of tube-level representations of flow cytometry data using the [DINO algorithm](https://arxiv.org/abs/2304.07193). This repo contains an implementation of the DinoFlow algorithm, with some additional code for training classification heads in a supervised fashion. 

The algorithm learns to generate embeddings by sampling separate sets of events from the same tube, performing augmentations on the sampled events such as scaling, shifting, and adding gaussian noise to them, and sending them through parallel 'teacher' and 'student' networks. The loss function is a cross entropy loss on the student and teacher outputs, and only the student parameters are updated. The teacher parameters are a slowly moving average (EMA) of the student

### Dependencies and installation

DinoFlow is a python project with dependencies managed by [uv](https://docs.astral.sh/uv/). After cloning this repository, executing `uv sync` should download and install all required dependencies (listed in the `pyproject.toml`) file. 

### Input data format for DINO training

The current implementation expects flow data to be saved as pytorch tensor files in a single directory, with one
file per sample. The flow data itself should be stored as a 2-dimensional tensor with each row a single event and the columns representing the markers, and should be stored under a single key inside the file. For instance, the following code should return a 2D tensor of marker intensities:

    torch.load("flow_data.pt", weights_only=False)['key']

For our experiments, the raw data were filtered to remove extremely bright events, compensated, and then $arcsinh$-transformed.


### Training a single tube 'backbone' with DinoFlow

Training requires a configuration file that specifies paths to training data (directory of .pt files with tube data), model parameters, and training parameters. An example can be found in the `conf.yaml` file in the repository root. To begin training with a single GPU, the command should resemble:

    uv run src/dinoflow/dino.py /path/to/conf.yaml --tube-type "data_key" --run-name "my_test_run"

The `--tube-type` parameter is the key where the raw data can be found in each sample .pt file. The `--run-name` parameter should be a unique name for this run, data are stored in a new folder with this name. 

DinoFlow supports distributed training, although it has only been tested in multi-GPU / single-node configuration. To begin a distributed training run (single node), a basic command might look like:

    torchrun --standalone --nnodes=1 --nproc-per-node=2  src/dinoflow/dino.py /path/to/conf.yaml ...

The `run_scripts/run_train.sh` script for an example of how to run dinoflow to train a new model.

Training configuration is specified through a configuration yaml - see `conf.yaml` for an example. 



### Supervised fine-tuning


There are multiple ways to evaluate, all of which involve training a 'classifier' head of some sort.
The simplest thing to do is use a single tube checkpoint (produced by `run_train.sh`) as a model. For instance, a single-tube model (with the correct markers) be use to predict AML, and any tube model can be used for viability or differentiating bone marrow / blood samples. 

Training and val input data are all provided by CSV files, which should contain labels and paths to tensor files that hold the raw tube event data for each sample. 

A full supervised fine-tuning command for a single tube looks like:

    uv run src/dinoflow/eval.py train \
        <run-name> \
        /path/to/training_samples_labels.csv \
        /path/to/validation_samples_labels.csv \
        /path/to/dinoflow_checkpoint.pt \
        conf.yaml \
        --labelkey 'AML' \       # Text key in label column to target
        --mode 'binary' \        # Binary prediction - True if label contains the labelkey, False otherwise
        --tube-type data_key \
        --freeze-backbone \
        --dataroot /directory/containing/flow/data/files/ \
        --events 4096 \
        --batch-size 128 \
        --epochs 50 \
        --comet-workspace "workspace" \   # If using comet to track experiments
        --comet-project "dinoflow" \


The training and validation CSVs must include a "path" and "label" column. The "path" column should be the path to a raw data .pt file (see 'Input Data Format' section), which can be absolute or relative to the "dataroot" argument. The label column can contain any value or text string. If --mode="regression", then label values are interpreted as regression targets. If --mode="binary", then labels are converted to boolean targets based on the "labelkey" argument. For instance, if labelkey is "CLL" and the label value is CLL, then the the label will be True for that sample, but if the label value is "AML" the value will be False. 
