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

in = load(file, "in")
trTs = load(file, "tr")


tr = vec(tr)
crossValidation = crossvalidation(tr, 5)





kernels = ["linear", "poly", "rbf", "sigmoid"]


# SVC
for kernel in kernels
    if kernel == "linear"
        for c in [0.5, 10.0]
            parameters = set_modelHyperparameters(kernel=kernel, C=c)

            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                precision, stdPrecision, negativePredictiveValues, _, f1, _, 
                matrix) = modelCrossValidation(:SVC, parameters, in, tr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
                negativePredictiveValues, f1, matrix)
        end
    elseif kernel == "poly"
        for params in [
            (2.0, 2, 0.1, 2.5),
            (2.0, 4, 1.2, 1.5)]
            (c, degree, gamma, coef0) = params
            parameters = set_modelHyperparameters(kernel=kernel, C=c, degree=degree, gamma=gamma, coef0=coef0)

            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                precision, stdPrecision, negativePredictiveValues, _, f1, _, 
                matrix) = modelCrossValidation(:SVC, parameters, in, tr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            println("\tDegree: ", degree)
            println("\tGamma: ", gamma)
            println("\tCoef0: ", coef0)
            _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
                negativePredictiveValues, f1, matrix)
        end
    elseif kernel == "rbf"
        for params in [
            (2.0, 0.9),
            (3.5, 0.2)]
            (c, gamma) = params
            parameters = set_modelHyperparameters(kernel=kernel, C=c, gamma=gamma)

            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                precision, stdPrecision, negativePredictiveValues, _, f1, _, 
                matrix) = modelCrossValidation(:SVC, parameters, in, tr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            println("\tGamma: ", gamma)
            _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
                negativePredictiveValues, f1, matrix)
        end
    elseif kernel == "sigmoid"
        for params in [
            (2.0, 1.3, 7.5),
            (1.5, 0.7, 8.5)]
            (c, gamma, coef0) = params
            parameters = set_modelHyperparameters(kernel=kernel, C=c, gamma=gamma, coef0=coef0)

            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                precision, stdPrecision, negativePredictiveValues, _, f1, _, 
                matrix) = modelCrossValidation(:SVC, parameters, in, tr, crossValidation)

            println("\nMetrics for SVC with kernel: ", kernel)
            println("Parameters:")
            println("\tC: ", c)
            println("\tGamma: ", gamma)
            println("\tCoef0: ", coef0)
            _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
                negativePredictiveValues, f1, matrix)
        end
    end
end

