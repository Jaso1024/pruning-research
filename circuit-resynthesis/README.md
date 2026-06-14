# Circuit Resynthesis Prototype

This is the small MNIST experiment for testing the "trained model -> circuit IR -> better substrate" idea before scaling it back to Depth Anything.

`mnist_circuit_resynthesis.py` trains a deliberately mismatched one-hidden-layer MLP on MNIST, probes whether FC1 neurons are local/image-like, and compares:

- no-retrain magnitude and WANDA pruning of the source MLP;
- local-window masks decompiled from source FC1 weights;
- random local masks at the same density;
- per-neuron top-k masks at the same density;
- tiny convolutional students initialized from recovered source patches;
- black-box distillation baselines.

The compact result folders under `outputs/` preserve the runs that mattered. The best full run in `outputs/masked_topk_v1/summary.md` showed local/top-k masked MLP resynthesis matching or slightly exceeding the dense source while using far fewer active weights, while the naive conv resynthesis did not beat plain distillation.
