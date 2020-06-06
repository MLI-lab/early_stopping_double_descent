# Early_stopping_double_descent
Code for reproducing figures and results in the paper ``Early stopping in deep networks: Double descent and how to eliminate it''

## Requirements
The following Python libraries are required to run the code in this repository:

```
numpy
torch
torchvision
```
and can be installed with `pip install -r requirements.txt`.

## Usage
The figures in the paper for the 5-layer convolutional network can be reproduced by running the `early_stopping_double_descent.ipynb` notebook. 

The numerical results can be reproduced by training the 5-layer convolutional network with `python3 train.py --config $CONFIG_FILE` where `CONFIG_FILE` points to the `config.json` file of the desired setup in the `./results/` directory.