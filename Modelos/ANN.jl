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

in = load(file, "in")
tr = load(file, "tr")

tr = vec(tr)
crossValidation = crossvalidation(tr, 5)

topologies = [[2], [4], [8], [10], [2 2], [2 4], [4 2], [4 4]]
# DecissionTreeClassifier
for topology in topologies
    parameters = set_modelHyperparameters(topology=topology)

    (acc, _, errorRate, _, sensibility, sensibilityStd, specificity, _,
        precision, precisionStd, negativePredictiveValues, _, f1, _, matrix) = modelCrossValidation(:ANN, parameters, in, tr, crossValidation)

    
    println("\nMetrics for DecisionTreeClassifier")
    println("Parameters:")
    println("\tMax depth: ", max_depth)
    _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
        negativePredictiveValues, f1, matrix)
end

