using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON
# using CuArrays

outfile = "mnist_denseFuzz.bson"

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9) |> gpu

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

dataset = repeated((X, Y), 100)
evalcb = () -> @show(loss(X, Y))
opt = ADAM()

## Begin training

@info("Beginning training loop...")
training_time = @elapsed Flux.train!(loss, params(model), dataset, opt, cb = throttle(evalcb, 10))


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
