""" Primera Aproximacion """

using JLD2
using Base: setindex, indexed_iterate
using FileIO
using Statistics
using Random
using DelimitedFiles;
using Flux;
using Flux.Losses;

include("fonts/boletin04.jl");
include("fonts/boletin05.jl");
include("fonts/boletin06.jl");

import .Metrics: confusionMatrix;
import .CrossValidation: crossvalidation;
import .ScikitModels: modelCrossValidation, set_modelHyperparameters;
include("metrics.jl")

# Generara una semilla aleatoria

Random.seed!(88008)

# Cargar datos y Extraer las características de esa aproximación 

file = "VH-VL.jld2"

inputs = load(file, "in")
targets = load(file, "tr")

targets = vec(targets)

crossValidationIndices = crossvalidation(targets, 5)

#Modelos a probar
models = ["ANN", "SVC", "KNN", "DecisionTree"]


# Llamar a la función modelCrossValidation para realizar validación cruzada con
# distintos modelos y configuraciones de parámetros. 

for model in models
    if(model=="ANN")
        topologies = [[2], [4], [8], [10], [2, 2], [2, 4], [4, 2], [4, 4]]
        for topology in topologies
            parameters = set_modelHyperparameters(topology=topology, numExecutions=20)
                
            (acc, _, errorRate, _, sensibility, sensibilityStd, specificity, _,
                precision, precisionStd, negativePredictiveValues, _, f1, _, matrix) =
                modelCrossValidation(:ANN, parameters, inputs, targets, crossValidationIndices)

            println("\nMetrics for ANN")
            println("Parameters:")
            println("\tTopology: ", topology)
            _show_metrics(acc, errorRate, sensibility, sensibilityStd, specificity, precision, precisionStd,
                negativePredictiveValues, f1, matrix)

        end

    elseif (model == "SVC")
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        for kernel in kernels
            if kernel == "linear"
                for c in [2.0, 5.6]
                    parameters = set_modelHyperparameters(kernel=kernel, C=c)
        
                    (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                        precision, stdPrecision, negativePredictiveValues, _, f1, _, 
                        matrix) = modelCrossValidation(:SVC, parameters, inputs, targets, crossValidationIndices)
                
                    println("\nMetrics for SVC with kernel: ", kernel)
                    println("Parameters:")
                    println("\tC: ", c)
                    _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
                        negativePredictiveValues, f1, matrix)
                end

            elseif kernel == "poly"
                for params in [
                    (2.0, 2, 0.1, 2.5),
                    (2.0, 3, 1.2, 1.5)]


                    (c, degree, gamma, coef0) = params
                    parameters = set_modelHyperparameters(kernel=kernel, C=c, degree=degree, gamma=gamma, coef0=coef0)

                    (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                        precision, stdPrecision, negativePredictiveValues, _, f1, _, 
                        matrix) = modelCrossValidation(:SVC, parameters, inputs, targets, crossValidationIndices)
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
                        matrix) = modelCrossValidation(:SVC, parameters, inputs, targets, crossValidationIndices)

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
                        matrix) = modelCrossValidation(:SVC, parameters, inputs, targets, crossValidationIndices)

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
    elseif (model == "KNN")
        for n_neighbors in [2, 4, 6, 8, 16, 24]
            parameters = set_modelHyperparameters(n_neighbors=n_neighbors)

            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                precision, stdPrecision, negativePredictiveValues, _, f1, _, matrix) = modelCrossValidation(:KNeighborsClassifier, parameters, inputs, targets, crossValidationIndices)

            println("\nMetrics for KNeighborsClassifier")
            println("Parameters:")
            println("\tn_neighbors: ", n_neighbors)
            _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
                negativePredictiveValues, f1, matrix)

        end

    elseif (model == "DecisionTree")
        for max_depth in [4, 6, 8, 12, 18, 24]
            parameters = set_modelHyperparameters(max_depth=max_depth)
          
            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
              precision, stdPrecision, negativePredictiveValues, _, f1, _, matrix) = modelCrossValidation(:DecissionTreeClassifier, parameters, inputs, targets, crossValidationIndices)
          
            println("\nMetrics for DecisionTreeClassifier")
            println("Parameters:")
            println("\tMax depth: ", max_depth)
          
            _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
              negativePredictiveValues, f1, matrix)
        end
    end
end