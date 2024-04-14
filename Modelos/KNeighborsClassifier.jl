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
include("../metrics.jl")

import .Metrics: confusionMatrix;
import .CrossValidation: crossvalidation;
import .ScikitModels: modelCrossValidation, set_modelHyperparameters;



file = "VH-VL.jld2"

in = load(file, "in")
tr = load(file, "tr")


tr = vec(tr)
crossValidation = crossvalidation(tr, 5)




# KNeighborsClassifier
for n_neighbors in [2, 4, 6, 8, 16, 24]
    parameters = set_modelHyperparameters(n_neighbors=n_neighbors)

    (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
        precision, stdPrecision, negativePredictiveValues, _, f1, _, matrix) = modelCrossValidation(:KNeighborsClassifier, parameters, in, tr, crossValidation)

    println("\nMetrics for KNeighborsClassifier")
    println("Parameters:")
    println("\tn_neighbors: ", n_neighbors)
    _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
        negativePredictiveValues, f1, matrix)
end
