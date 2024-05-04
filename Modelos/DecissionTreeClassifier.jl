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
include("../errorFunctions/errorFunctions.jl")

import .Metrics: confusionMatrix;
import .CrossValidation: crossvalidation;
import .ScikitModels: modelCrossValidation, set_modelHyperparameters;
import .ErrorFunctions: showErrorFunctions;

include("../metrics.jl")

Random.seed!(88008)

file = "VH-M-VL.jld2"

in = load(file, "in")
tr = load(file, "tr")

tr = vec(tr)
crossValidation = crossvalidation(tr, 5)

# DecissionTreeClassifier
for max_depth in [4, 6, 8, 12, 18, 24]
  parameters = set_modelHyperparameters(max_depth=max_depth)

  (acc, _, errorRate, _, sensibility, stdSensibility, specificity, _,
    precision, stdPrecision, negativePredictiveValues, _, f1, _, matrix, mse, mseD, mae, maeD, msle, msleD, rmse, rmseD) = modelCrossValidation(:DecissionTreeClassifier, parameters, in, tr, crossValidation, true)

  println("\nMetrics for DecisionTreeClassifier")
  println("Parameters:")
  println("\tMax depth: ", max_depth)

  println("mse: ", mse)
  println("mse (std): ", mseD)
  println("mae: ", mae)
  println("mae (std): ", maeD)
  println("msle: ", msle)
  println("msle (std): ", msleD)
  println("rmse: ", rmse)
  println("rmse (std): ", rmseD)

  _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision,
    negativePredictiveValues, f1, matrix)

end

