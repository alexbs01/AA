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

inTr = load(file, "inTr")
inVl = load(file, "inVl")
inTs = load(file, "inTs")
trTr = load(file, "trTr")
trVl = load(file, "trVl")
trTs = load(file, "trTs")


trTr = vec(trTr)
crossValidation = crossvalidation(trTr, 5)




# KNeighborsClassifier
for n_neighbors in [3, 5, 7, 9, 11]
    parameters = set_modelHyperparameters(n_neighbors=n_neighbors)

    (acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
            negativePredictiveValues, f1, matrix) = modelCrossValidation(:KNeighborsClassifier, parameters, inTr, trTr, crossValidation)

    println("\nMetrics for KNeighborsClassifier")
    println("Parameters:")
    println("\tn_neighbors: ", n_neighbors)
    _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
        negativePredictiveValues, f1, matrix)
end
