module ANN

using JLD2
using Base: setindex, indexed_iterate
using FileIO
using Statistics
using Random
using DelimitedFiles
using Flux
using Flux.Losses

export execute

include("../fonts/boletin04.jl")
include("../fonts/boletin05.jl")
include("../fonts/boletin06.jl")

import .Metrics: show_metrics
import .CrossValidation: crossvalidation
import .ScikitModels: modelCrossValidation, set_modelHyperparameters

# Load data

function execute(file::String, regresion::Bool=true)

    in = load(file, "in")
    tr = load(file, "tr")

    tr = vec(tr)
    crossValidation = crossvalidation(tr, 5)

    topologies = [[2], [4], [8], [10], [2, 2], [2, 4], [4, 2], [4, 4]]
    # DecissionTreeClassifier
    for topology in topologies
        parameters = set_modelHyperparameters(topology=topology)


        if (regresion)
            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                precision, stdPrecision, negativePredictiveValues, _, f1, _, matrix,
                mse, mseD, mae, maeD, msle, msleD, rmse, rmseD) =
                modelCrossValidation(:ANN, parameters, in, tr, crossValidation, regresion)

            println("\nMetrics for ANN")
            println("Parameters:")
            println("\tTopology: ", topology)
            println("mse: ", mse)
            println("mse (std): ", mseD)
            println("mae: ", mae)
            println("mae (std): ", maeD)
            println("msle: ", msle)
            println("msle (std): ", msleD)
            println("rmse: ", rmse)
            println("rmse (std): ", rmseD)

        else
            (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
                precision, stdPrecision, negativePredictiveValues, _, f1, _, matrix) =
                modelCrossValidation(:ANN, parameters, in, tr, crossValidation, regresion)


        end





        show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
            negativePredictiveValues, f1, matrix)
    end
end

end

