# cd("path/to/repository")

# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()

using BSON

data_conv = BSON.load("mnist_conv.bson")
data_convFuzz = BSON.load("mnist_convFuzz.bson")
data_convAdpt = BSON.load("mnist_convAdpt.bson")
data_convFull = BSON.load("mnist_convFull.bson")

data_dense = BSON.load("mnist_dense.bson")
data_denseFuzz = BSON.load("mnist_denseFuzz.bson")
data_denseAdpt = BSON.load("mnist_denseAdpt.bson")
data_denseFull = BSON.load("mnist_denseFull.bson")

compute_testing_speed(data) = 10_000 / data[:testing_time]
compute_testing_efficiency(data) = data[:testing_accuracy] * compute_testing_speed(data)
compute_testing_efficiency_fuzzed(data) = data[:testing_accuracy_fuzzed] * compute_testing_speed(data)

compute_testing_speed(data_conv)
compute_testing_efficiency(data_conv)
