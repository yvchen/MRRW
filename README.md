## MRRW (Two-Layer Mutually Reinforced Random Walk)

Yun-Nung (Vivian) Chen, yvchen@cs.cmu.edu

This tool is used to compute the converged scores of nodes in a two-layer graph.
Given within-layer and between-layer edge weights, the score for each node refers to its importance based on the structure.

### Requirements
1. Python 2.7
2. Numpy

### Input data
1. Edge weights for the 1st layer (layer1-edge-weight)
2. Edge weights for the 2nd layer (layer2-edge-weight)
3. Edge weights between two layers (layer1to2-edge-weight)
4. (Optional) Initial node scores in the 1st layer (layer1-init-score)
5. (Optional) Initial node scores in the 2st layer (layer2-init-score)

Each file should have a matrix form as follows (space delimited).

```0.7 0.2 0.5 ...```

### Running the program

    python program/twolayer.py \
    layer1-edge-weight layer2-edge-weight layer1to2-edge-weight \
    [--initscore1 layer1-init-score] [--initscore2 layer2-init-score] [-w alpha] [-n N]
    
You can easily run the testing examples as follows:

    python program/twolayer.py example/w1.txt example/w2.txt example/w3.txt

### Reference

Main paper to be cited
```
@Inproceedings{chen:2015:NAACL,
  author    = {Chen, Yun-Nung and Wang, William Yang and Rudnicky, Alexander I.},
  title     = {Jointly Modeling Inter-Slot Relations by Random Walk on Knowledge Graphs for Unsupervised Spoken Language Understanding},
  booktitle = {Proceedings of NAACL},
  year      = {2015},
}
```
