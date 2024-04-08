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

inTr = load("VH-VL.jld2", "inTr")
inVl = load("VH-VL.jld2", "inVl")
inTs = load("VH-VL.jld2", "inTs")
trTr = load("VH-VL.jld2", "trTr")
trVl = load("VH-VL.jld2", "trVl")
trTs = load("VH-VL.jld2", "trTs")


trTr = vec(trTr)
crossValidation = crossvalidation(trTr, 5)





kernels = ["linear", "poly", "rbf", "sigmoid"]


# SVC
for kernel in kernels
    if kernel == "linear"
        for c in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            parameters = set_modelHyperparameters(kernel=kernel, C=c)

            (acc, errorRate, sensibility, specificity, precision,
                negativePredictiveValues, f1,
                matrix) = modelCrossValidation(:SVC, parameters, inTr, trTr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            _show_metrics(acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)
        end
    elseif kernel == "poly"
        for params in [(1.0, 1, 0.8, 1.5),
            (2.0, 2, 0.1, 2.5),
            (2.0, 3, 0.4, 1.5),
            (2.0, 4, 1.2, 1.5),
            (2.0, 3, 1.5, 1.5)]
            (c, degree, gamma, coef0) = params
            parameters = set_modelHyperparameters(kernel=kernel, C=c, degree=degree, gamma=gamma, coef0=coef0)

            (acc, errorRate, sensibility, specificity, precision,
                negativePredictiveValues, f1,
                matrix) = modelCrossValidation(:SVC, parameters, inTr, trTr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            println("\tDegree: ", degree)
            println("\tGamma: ", gamma)
            println("\tCoef0: ", coef0)
            _show_metrics(acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)
        end
    elseif kernel == "rbf"
        for params in [(1.0, 1.0),
            (2.0, 0.7),
            (3.0, 0.3),
            (2.0, 0.1),
            (3.5, 0.5)]
            (c, gamma) = params
            parameters = set_modelHyperparameters(kernel=kernel, C=c, gamma=gamma)

            (acc, errorRate, sensibility, specificity, precision,
                negativePredictiveValues, f1,
                matrix) = modelCrossValidation(:SVC, parameters, inTr, trTr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            println("\tGamma: ", gamma)
            _show_metrics(acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)
        end
    elseif kernel == "sigmoid"
        for params in [(1.0, 1.3, 1.0),
            (1.5, 0.5, 3.5),
            (2.0, 1.3, 7.5),
            (1.2, 0.2, 8.0),
            (1.5, 0.7, 1.5)]
            (c, gamma, coef0) = params
            parameters = set_modelHyperparameters(kernel=kernel, C=c, gamma=gamma, coef0=coef0)

            (acc, errorRate, sensibility, specificity, precision,
                negativePredictiveValues, f1,
                matrix) = modelCrossValidation(:SVC, parameters, inTr, trTr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            println("\tGamma: ", gamma)
            println("\tCoef0: ", coef0)
            _show_metrics(acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)
        end
    end
end
