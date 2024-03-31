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


include("boletines/boletin02.jl");
include("boletines/boletin03.jl");
include("boletines/boletin04.jl");
include("boletines/boletin05.jl");
include("boletines/boletin06.jl");
import .ANNUtils: oneHotEncoding, calculateMinMaxNormalizationParameters, 
                calculateZeroMeanNormalizationParameters, normalizeMinMax,
                normalizeMinMax!, normalizeZeroMean, normalizeZeroMean!, 
                classifyOutputs, accuracy, buildClassANN, trainClassANN;
                
import .Overtraining: holdOut;
import .Metrics: confusionMatrix, printConfusionMatrix;
import .CrossValidation: crossvalidation, ANNCrossValidation;
import .ScikitModels: modelCrossValidation, set_modelHyperparameters;

inTr = load("VH-VL.jld2", "inTr")
inVl = load("VH-VL.jld2", "inVl")
inTs = load("VH-VL.jld2", "inTs")
trTr = load("VH-VL.jld2", "trTr")
trVl = load("VH-VL.jld2", "trVl")
trTs = load("VH-VL.jld2", "trTs")

parameters = set_modelHyperparameters(:SVC, topology=[2, 2], maxEpochs=100, C=2.0, kernel="sigmoid", degree=3, gamma=3.0 , coef0=5.0)

println(parameters)

trTr = vec(trTr)
crossValidation = crossvalidation(trTr, 5)
println(typeof(trTr))
println(size(trTr))
(acc, errorRate, sensibility, specificity, precision, 
    negativePredictiveValues, f1, 
    matrix) = modelCrossValidation(:SVC, parameters, inTr, trTr, crossValidation)

println("Metrics for ANN")
println("Accuracy: ", acc)
println("Error rate: ", errorRate)
println("Sensibility: ", sensibility)
println("Specificity: ", specificity)
println("Precision: ", precision)
println("Negative predictive values: ", negativePredictiveValues)
println("F1: ", f1)
println("Confusion matrix: ")
println(matrix)