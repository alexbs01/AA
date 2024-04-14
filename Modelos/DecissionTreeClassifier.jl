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

file = "VH-VL.jld2"

inTr = load(file, "inTr")
inVl = load(file, "inVl")
inTs = load(file, "inTs")
trTr = load(file, "trTr")
trVl = load(file, "trVl")
trTs = load(file, "trTs")


trTr = vec(trTr)
crossValidation = crossvalidation(trTr, 5)




# DecissionTreeClassifier
for max_depth in [3, 6, 12, 18, 24]
    parameters = set_modelHyperparameters(max_depth=max_depth)

    (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
        precision, stdPrecision, negativePredictiveValues, _, f1, _, matrix) = modelCrossValidation(:DecissionTreeClassifier, parameters, inTr, trTr, crossValidation)

    println("\nMetrics for DecisionTreeClassifier")
    println("Parameters:")
    println("\tMax depth: ", max_depth)
    _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
        negativePredictiveValues, f1, matrix)
end

