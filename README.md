# A Perspective on Memory-Associated Differential Learning

## Install Enviornment:
```
    pip install -r requirements.txt
```

## Running a model
We have streamlined our code so that you can run any of five models on the OGBL-DDI dataset. Simply run: ```python ogbl-script.py``` to train our MAD + GCN model to 100 epochs, and run ```python ogbl-script.py --model MAD_SAGE``` to train the MAD + GraphSAGE model to 100 epochs.