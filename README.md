# Factorized Gaussian Process VAE

Code for paper Factorized Gaussian Process Variational Autoencoders. 

Initially forked from [this cool repo](https://github.com/scrambledpie/GPVAE).

## Dependencies
* Python >= 3.6
* TensorFlow = 1.15
* TensorFlow Probability = 0.8

## Setup 
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install dependencies, using `pip install -r requirements.txt`
4. Test the setup by running `python BALL_experiment.py --elbo VAE`
## Experiments

To produce results presented in the paper run `python FGPVAE_experiments.py --GECO --kappa_squared 0.020`.

For all available configurations run `python --FGPVAE_experiment.py --help`

To generate other rotated MNIST datasets use `generate_rotated_MNIST` function in `utils.py`.

Implementation of baselines (GP-VAE, CVAE) can be found [here](https://github.com/ratschlab/SVGP-VAE).

## Authors
- Metod Jazbec (jazbec.metod@gmail.com)

