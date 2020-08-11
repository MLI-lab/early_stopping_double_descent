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

**Figure 3**: Two layer neural network kernels and risk can be reproduced by running the `two_layer_nn` notebook.
    
**Figure 4**: Double descent in the two layer neural network and the elimination of the double descent through the scaling of the stepsizes of the two layers can be reproduced by running the `early_stopping_two-layer-nn_double_descent.ipynb` notebook. 

**Figure 1-a, 5**: Double descent in the 5-layer convolutional network and the elimination of the double descent through the scaling of the stepsizes of the different layers can be reproduced by running the `early_stopping_double_descent.ipynb` notebook. 

The numerical results can be reproduced by training the 5-layer convolutional network with `python3 train.py --config $CONFIG_FILE` where `CONFIG_FILE` points to the `config.json` file of the desired setup in the `./results/` directory.

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