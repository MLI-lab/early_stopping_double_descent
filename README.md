# Early_stopping_double_descent
This repository contains the code for reproducing figures and results in the paper ``Early Stopping in Deep Networks: Double Descent and How to Eliminate it''.

# Requirements
The following Python libraries are required to run the code in this repository:

```
numpy
jupyter
torch
torchvision
```
and can be installed with `pip install -r requirements.txt`.


# Usage
All the figures in the paper can be reproduced by running the respective notebooks as indicated below:

**Figure 2**: Bias-variance trade-off curves for the linear regression model can be reproduced by running the `sum_bias_variance_tradeoffs` notebook.

**Figure 3**: Double descent in the two layer neural network and the elimination of the double descent through the scaling of the stepsizes of the two layers can be reproduced by running the `early_stopping_two-layer-nn_double_descent.ipynb` notebook.

**Figure 1-a, 4**: Double descent in the 5-layer convolutional network and the elimination of the double descent through the scaling of the stepsizes of the different layers can be reproduced by running the `early_stopping_deep_double_descent.ipynb` notebook. 

The numerical results can be reproduced by training the 5-layer convolutional network with `python3 train.py --config $CONFIG_FILE` where `CONFIG_FILE` points to the `config.json` file of the desired setup in the `./results/` directory.

* NOTE: * Please set the param `gpu` in the configs accordingly. Mostly it will be `0` but you might get errors with the default setting. Please change it accrodingly.

## Disclaimers
**Figure 1-a, 7**: The bias and variance is measured as proposed in [Yang et al. \[2020\]](https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff) but adopted to measure bias-variance at each epoch. This may result in highly noisy measurements for the early training phase (see [this notebook](notebooks/early_stopping_deep_double_descent.ipynb) for details).

## Citation
```
@article{heckel_yilmaz_2020,
    author    = {Reinhard Heckel and Fatih Furkan Yilmaz},
    title     = {Early Stopping in Deep Networks: Double Descent and How to Eliminate it},
    journal   = {arXiv:2007.10099},
    year      = {2020}
}
```

## Licence

All files are provided under the terms of the Apache License, Version 2.0.
