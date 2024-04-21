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

file = "VH-M-VL.jld2"

in = load(file, "in")
tr = load(file, "tr")

tr = vec(tr)
crossValidation = crossvalidation(tr, 5)

# DecissionTreeClassifier
for max_depth in [4, 6, 8, 12, 18, 24]
  parameters = set_modelHyperparameters(max_depth=max_depth)

  (mse, mseD, mae, maeD, msle, msleD, rmse, rmseD) = modelCrossValidation(:DecissionTreeClassifier, parameters, in, tr, crossValidation)

  println("\nMetrics for DecisionTreeClassifier")
  println("Parameters:")
  println("\tMax depth: ", max_depth)
  
  showErrorFunctions(mse, mae, msle, rmse)

end

