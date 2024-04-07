using Pkg;
"""Pkg.add("DelimitedFiles");
Pkg.add("Flux");
Pkg.add("Statistics");"""

using JLD2
using Base: setindex, indexed_iterate
using FileIO
using Statistics
using Random
using DelimitedFiles;
using Flux;
using Flux.Losses;

include("fonts/boletin04.jl")
include("fonts/boletin05.jl")
include("fonts/boletin06.jl");

import .Metrics: confusionMatrix;
import .CrossValidation: crossvalidation;
import .ScikitModels: modelCrossValidation, set_modelHyperparameters;

function _print_matrix(matrix)
    rows, cols = size(matrix)
    for i in 1:rows
        for j in 1:cols
            print(matrix[i, j], "\t")
        end
        println()
    end
end

function _show_metrics(acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)
    println("Accuracy: ", acc)
    println("Error rate: ", errorRate)
    println("Sensibility: ", sensibility)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("Negative predictive values: ", negativePredictiveValues)
    println("F1: ", f1)
    println("Confusion matrix: ")
    _print_matrix(matrix)
end

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
        for params in [ (1.0, 1, 0.8, 1.5), 
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

# DecissionTreeClassifier
for max_depth in [3, 6, 12, 18, 24]
    parameters = set_modelHyperparameters(max_depth=max_depth)

    (acc, errorRate, sensibility, specificity, precision, 
    negativePredictiveValues, f1, 
    matrix) = modelCrossValidation(:DecissionTreeClassifier, parameters, inTr, trTr, crossValidation)

    println("\nMetrics for DecisionTreeClassifier")
    println("Parameters:")
    println("\tMax depth: ", max_depth)
    _show_metrics(acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)
end

# KNeighborsClassifier
for n_neighbors in [3, 5, 7, 9, 11]
    parameters = set_modelHyperparameters(n_neighbors=n_neighbors)

    (acc, errorRate, sensibility, specificity, precision, 
    negativePredictiveValues, f1, 
    matrix) = modelCrossValidation(:KNeighborsClassifier, parameters, inTr, trTr, crossValidation)

    println("\nMetrics for KNeighborsClassifier")
    println("Parameters:")
    println("\tn_neighbors: ", n_neighbors)
    _show_metrics(acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)
end