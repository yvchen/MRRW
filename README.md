## MRRW (Two-Layer Mutually Reinforced Random Walk)

Yun-Nung (Vivian) Chen, yvchen@cs.cmu.edu

This tool is used to compute the converged scores of nodes in a two-layer graph.
Given within-layer and between-layer edge weights, the score for each node refers to its importance based on the structure.

### Requirements
1. Python 2.7
2. Numpy
3. Scipy

### Input data
1. Edge weights for the 1st layer (layer1-edge-weight)
2. Edge weights for the 2nd layer (layer2-edge-weight)
3. Edge weights between two layers (layer1to2-edge-weight)
4. (Optional) Initial node scores in the 1st layer (--initscore1 layer1-init-score)
5. (Optional) Initial node scores in the 2st layer (--initscore2 layer2-init-score)

### Parameter setting
1. -w: Interpolation weight for the propagation part (default=0.9)
2. -n: Keep only within-layer edges with top N highest weights (default mode is to keep all edges)
3. -m: Keep only between-layer edges with top M highest weights (default mode is to keep all edges)
4. -s: flag to indicate whether your input matrices are in the sparse format (default=0).

### Data format
##### Input matrix
Each file should have a dense matrix form as follows (space delimited).

```0.7 0.2 0.5 ...```

For a sparse matrix, the format should be as follows (space delimited).

    2 3 2
    0 2 .1
    1 1 .3

This refers to a 2x3 matrix and there are 2 nonzero items (first line in the file).
The items are M(0, 2)=0.1 and M(1, 1)=0.3.

##### Output scores
The program outputs the converged score for each node in one line.
Small examples can be found in ```output/```.

### Running the program

    python program/twolayer.py \
    layer1-edge-weight layer2-edge-weight layer1to2-edge-weight \
    [--initscore1 layer1-init-score] [--initscore2 layer2-init-score] [-w alpha] [-n N] [-m M] [-s sparse] \
    [--outfile1 layer1-fin-score] [--outfile2 layer2-fin-score] 

You can easily run the testing example as follows:

    python program/twolayer.py example/w1.txt example/w2.txt example/w3.txt
    
Also, this is another testing example with sparse format and more defined paramters:

    python program/twolayer.py example/w1-s.txt example/w2-s.txt example/w3-s.txt -w 0.8 -n 2 -m 2 --outfile1 output/s1.txt --outfile2 output/s2.txt -s 1

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
