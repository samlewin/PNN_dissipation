# Probablistic neural networks for predicting energy dissipation rates in geophysical turbulent flows

This repository is the official implementation of the paper `Probablistic neural networks for predicting energy dissipation rates in geophysical turbulent flows'

## Requirements

The code included in this repository is set up for use on a GPU enabled with CUDA and CuDNN. To install the required packages, run the following command:

```setup
pip install -r requirements.txt
```

We recommend using a [Python 3 virtual environment](https://docs.python.org/3/library/venv.html).

## Training

Note that, due to large file sizes, training data is available upon request. However, testing data and a pre-trained model which can be used together to reproduce the figures in the paper are included. To train the PNN described in the paper, run the following command:

```train
python3 train_model.py 
```

This will create a file called 'model_modelname.h5' where modelname is the name specified in train_model.py. 

## Testing

To test the model on the included test dataset, run:

```eval
python test_model.py 
```

## Evaluation code

After testing as above, code for reproducing the figures found in the paper is available in the included jupyter notebook post_processing.ipynb.

## Pre-trained Models

A pre-trained model model_shearr_mixed_all.h5 is included in the repository. This will be loaded by default by test_model.py.

## Contributing

If you have any questions about the code, or any suggestions for improvements, please email *anonymized*, or open an issue on this repository. All content in this repository is licensed under the MIT license.

