
## DinoFlow - using DINO algorithm for Flow data

This repo holds experimental research code for using the DINO algorithm to generate useful "bag" (or "tube")-level embeddings of flow cytometry data

The algorithm learns to generate embeddings by sampling separate sets of events from the same tube, performing augmentations on the sampled events such as scaling, shifting, and adding gaussian noise to them, and sending them through parallel 'teacher' and 'student' networks. The loss function is a cross entropy loss on the student and teacher outputs, and only the student parameters are updated. The teacher parameters are a slowly moving average (EMA) of the student

### Training a single tube 'backbone' with dinoflow

Check out the `run_train.sh` script for an example of how to run dinoflow to train a new model. The model takes an arbitrary number of events and compresses them into a single vector

Training configuration is specified through a configuration yaml - see `conf.yaml` for an example. 

### Evaluation / Training classifiers

There are multiple ways to evaluate, all of which involve training a classifier of some sort.
The simplest thing to do is just use a single tube checkpoint (produced by run_train.sh) as a model. For instance, an m-tube model
can be use to predict AML, and any tube model can be used for viability or differentiating bone marrow / blood samples.
Evaluating single tube models can be accomplished by running the `run_eval.sh` script.

For more sophisticated analyses, you probably want to use data from all three tubes. If you have three tube model checkpoints, you can run the `run_eval3tubes.sh` script to train a classifier using all of them. The emits a new kind of checkpoint that contains all the tube backbones as well as the trained classifier head. 

To continue training a combined 3-tube model, run the `run_eval_trainfull.sh` script. You can optionally choose to unfreeze a certain number of backbone layers. 

Training and val input data are all provided by CSV files, which should contain labels and paths to tensor files that hold the raw tube event data for each sample. 



### Installing torch-cluster

torch-cluster is required to run some of the evaluation models, and cannot be installed automatically with uv. Instead, try this command after installing everything else:

    uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
