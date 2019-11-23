using Pkg
Pkg.activate("/home/alec/code/homework/CAS-CS-640/project/julia")
Pkg.instantiate()

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON
# using CuArrays

outfile = "mnist_denseFull.bson"

# Classify MNIST digits with a simple multi-layer-perceptron

## Preprocess the data

function preprocess(imgs)
    # return the images as a single matrix of features × samples
    return hcat(float.(reshape.(imgs, :))...) |> gpu
end

function make_minibatch(X, Y, idxs)
    # collect the inputs and the labels in a tuple
    # group as minibatches
    X_batch = Array{Float32}(undef, size(X, 1), length(idxs))
    for i in 1:length(idxs)
        X_batch[:, i] = Float32.(X[:, i])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9) |> gpu
    return (X_batch, Y_batch)
end

# load the MNIST dataset
training_images = MNIST.images() |> preprocess
training_labels = MNIST.labels()
# partition the training data into batches
batch_size = 128
mb_idxs = partition(1:size(training_images, 2), batch_size)
training_set = [make_minibatch(X, training_labels, i) for i in mb_idxs]

# construct the testing set as one giant minibatch
testing_images = MNIST.images(:test) |> preprocess
testing_labels = MNIST.labels(:test)
testing_set = make_minibatch(testing_images, testing_labels, 1:size(testing_images, 2))

## Construct the model

model = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

function loss(x, y)
  x_aug = x .+ 0.1f0 * gpu(randn(eltype(x), size(x)))
  y_hat = model(x_aug)
  return crossentropy(y_hat, y)
end

accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

opt = ADAM(0.001)

## Begin training

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
training_time = @elapsed for epoch_idx in 1:20
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), training_set, opt)

    # Calculate accuracy:
    acc = accuracy(testing_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))

    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy!")
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end


## Compute the metrics

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

testing_time = @elapsed model(tX)
testing_accuracy = accuracy(tX, tY)

function fuzz(x)
    return x .+ 0.1f0 * gpu(randn(eltype(x), size(x)))
end

testing_accuracy_fuzzed = accuracy(fuzz(tX), tY)

## Save the results

BSON.@save joinpath(pwd(), outfile) training_time testing_time testing_accuracy testing_accuracy_fuzzed
