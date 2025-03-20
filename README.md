
## DinoFlow - using DINO algorithm for Flow data

This repo holds experimental research code for using the DINO algorithm to generate useful "bag" (or "tube")-level embeddings of flow cytometry data

The algorithm learns to generate embeddings by sampling separate sets of events from the same tube, performing augmentations on the sampled events such as scaling, shifting, and adding gaussian noise to them, and sending them through parallel 'teacher' and 'student' networks. The loss function is a cross entropy loss on the student and teacher outputs, and only the student parameters are updated. The teacher parameters are a slowly moving average (EMA) of the student

Currently, only some experimental training code exists - see the `example_training.sh` script to see how to execute a training run.

Training configuration is specified through a configuration yaml - see `conf.yaml` for an example there. 
