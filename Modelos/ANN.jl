using JLD2
using Base: setindex, indexed_iterate
using FileIO
using Statistics
using Random
using DelimitedFiles;
using Flux;
using Flux.Losses;

include("../fonts/boletin04.jl")
include("../fonts/boletin05.jl")
include("../fonts/boletin06.jl");

import .Metrics: confusionMatrix;
import .CrossValidation: crossvalidation;
import .ScikitModels: modelCrossValidation, set_modelHyperparameters;

include("../metrics.jl")

# Load data

file = "VH-VL.jld2"

inTr = load(file, "inTr")
inVl = load(file, "inVl")
inTs = load(file, "inTs")
trTr = load(file, "trTr")
trVl = load(file, "trVl")
trTs = load(file, "trTs")

trTr = vec(trTr)
crossValidation = crossvalidation(trTr, 5)

topologies = [[2], [4], [8], [10], [2 2], [2 4], [4 2], [4 4]]
# DecissionTreeClassifier
for topology in topologies
    parameters = set_modelHyperparameters(topology=topology)

    (acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
        negativePredictiveValues, f1, matrix) = modelCrossValidation(:ANN, parameters, inTr, trTr, crossValidation)

    println("\nMetrics for DecisionTreeClassifier")
    println("Parameters:")
    println("\tMax depth: ", max_depth)
    _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
        negativePredictiveValues, f1, matrix)
end

