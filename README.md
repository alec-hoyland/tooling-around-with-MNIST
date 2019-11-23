# tooling-around-with-MNIST
CNN and feedforward NN model benchmarking in Julia on the MNIST dataset

Here are some convolutional neural network and feedforward fully-connected network models
tested on the MNIST handwritten digit dataset.
The models use the [Flux](https://github.com/FluxML/Flux.jl) machine learning package.

## Getting started

To create an environment for the repository,
navigate to the repository directory, then in Julia:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then, any of the models can be run normally. The models are:

* `Conv`, the base convolutional neural network
* `ConvFuzz`, CNN with input fuzzing
* `ConvAdpt`, CNN with adaptive learning rate
* `ConvFull`, CNN with both input fuzzing and adaptive learning rate
* `Dense`, the base fully-connected feedforward neural network
* `DenseFuzz`, feedforward network with input fuzzing
* `DenseAdpt`, feedforward network with adaptive learning rate
* `DenseFull`, feedforward network with both input fuzzing and adaptive learning rate

The models will run fine on the CPU.
They should also run on CUDA-capable GPUs.
To do this, use the `CuArrays` package,

```julia
using CuArrays
```

then run the models normally.

## The paper

There is a short paper detailing results [here](https://github.com/alec-hoyland/tooling-around-with-MNIST/blob/master/paper/nips_2018.pdf).
