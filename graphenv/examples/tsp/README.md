# Traveling Salesperson Problem (TSP)

Demo of using graphenv + networkx to define a TSP problem on a 
fully connected planar graph, i.e., where node features are 
x-y coordinates and edge weights are pairwise distances.

To run a simple example that creates the graph, solves TSP using
networkx TSP heuristics, and compares with the TSPEnv with random
actions:

```
python tsp_env.py
```

You can change the size and seed of the networkx graph, see:

```
python tsp_env.py --help
```

The `tsp-env.ipynb` notebook is effectively the same as this demo
but with network viz.

A script for solving the TSPEnv with RLLib is available in 
`graph-env/docs/examples/run_tsp.py`.



