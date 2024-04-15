module ScikitModels
using Pkg
"""Pkg.add("Statistics")"""

include("boletin02.jl")
include("boletin03.jl")
include("boletin04.jl")
include("boletin05.jl")
using ScikitLearn
using ScikitLearn: fit!, predict
using Flux
using Flux.Losses
using Statistics
using .Metrics: confusionMatrix
using .ANNUtils: oneHotEncoding, trainClassANN
using .Overtraining: holdOut
using .CrossValidation: ANNCrossValidation

export modelCrossValidation, set_modelHyperparameters

@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
  inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
  crossValidationIndices::Array{Int64,1})

  @assert size(inputs, 1) == size(targets, 1) "Inputs and targets must have the same number of samples"
  @assert (modelType == :SVC) ||
          (modelType == :DecissionTreeClassifier) ||
          (modelType == :KNeighborsClassifier) ||
          (modelType == :ANN) "Model must be SVC, DecissionTreeClassifier, KNeighborsClassifier or ANN"

  if modelType == :ANN
    topology = modelHyperparameters["topology"]
    maxEpochs = modelHyperparameters["maxEpochs"]
    maxEpochsVal = modelHyperparameters["maxEpochsVal"]
    learningRate = modelHyperparameters["learningRate"]
    transferFunctions = modelHyperparameters["transferFunctions"]
    validationRatio = modelHyperparameters["validationRatio"]
    minLoss = modelHyperparameters["minLoss"]
    numExecutions = modelHyperparameters["numExecutions"]

    (acc, accStd), (errorRate, errorRateStd), (sensibility, sensibilityStd), (specificity, specificityStd),
    (precision, precisionStd), (negativePredictiveValues, negativePredictiveValuesStd), (f1, f1Std), matrix =
      ANNCrossValidation(topology, inputs, targets, crossValidationIndices, numExecutions=numExecutions,
        transferFunctions=transferFunctions, maxEpochs=maxEpochs, learningRate=learningRate,
        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)

  else
    numExecutions = modelHyperparameters["numExecutions"]

    accuracyPerTraining = zeros(numExecutions)
    errorRatePerTraining = zeros(numExecutions)
    sensibilityPerTraining = zeros(numExecutions)
    specificityPerTraining = zeros(numExecutions)
    precisionPerTraining = zeros(numExecutions)
    negativePredictiveValuesPerTraining = zeros(numExecutions)
    f1PerTraining = zeros(numExecutions)
    confusionMatrixPerTraining = zeros(size(oneHotEncoding(targets), 2), size(oneHotEncoding(targets), 2), numExecutions)
    confusionMatrixPerTraining = zeros(2, 2, numExecutions)

    for numExecution in 1:numExecutions

      if modelType == :SVC
        possibleKernel = ["linear", "poly", "rbf", "sigmoid"]
        C = modelHyperparameters["C"]
        kernel = modelHyperparameters["kernel"]
        degree = modelHyperparameters["degree"]
        gamma = modelHyperparameters["gamma"]
        coef0 = modelHyperparameters["coef0"]
        @assert kernel in possibleKernel "Kernel must be linear, poly, rbf or sigmoid"

        if kernel == "linear"
          model = SVC(kernel=kernel, C=C)

        elseif kernel == "poly"
          model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0)

        elseif kernel == "rbf"
          model = SVC(kernel=kernel, C=C, gamma=gamma)

        elseif kernel == "sigmoid"
          model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0)

        end

      elseif modelType == :DecissionTreeClassifier
        max_depth = modelHyperparameters["max_depth"]

        model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)

      elseif modelType == :KNeighborsClassifier
        n_neighbors = modelHyperparameters["n_neighbors"]

        model = KNeighborsClassifier(n_neighbors=n_neighbors)
      end

      trainIndex = findall(crossValidationIndices .!= 1)
      testIndex = findall(crossValidationIndices .== 1)

      trainingInputs = inputs[trainIndex, :]
      trainingTargets = targets[trainIndex]
      testingInputs = inputs[testIndex, :]
      testingTargets = targets[testIndex]

      model = fit!(model, trainingInputs, trainingTargets)

      outputs = predict(model, testingInputs)

      (
        accuracyPerTraining[numExecution],
        errorRatePerTraining[numExecution],
        sensibilityPerTraining[numExecution],
        specificityPerTraining[numExecution],
        precisionPerTraining[numExecution],
        negativePredictiveValuesPerTraining[numExecution],
        f1PerTraining[numExecution],
        confusionMatrixPerTraining[:, :, numExecution],
			) = confusionMatrix(outputs, testingTargets)

    end

    acc = mean(accuracyPerTraining)
    accStd = std(accuracyPerTraining)
    errorRate = mean(errorRatePerTraining)
    errorRateStd = std(errorRatePerTraining)
    sensibility = mean(sensibilityPerTraining)
    sensibilityStd = std(sensibilityPerTraining)
    specificity = mean(specificityPerTraining)
    specificityStd = std(specificityPerTraining)
    precision = mean(precisionPerTraining)
    precisionStd = std(precisionPerTraining)
    negativePredictiveValues = mean(negativePredictiveValuesPerTraining)
    negativePredictiveValuesStd = std(negativePredictiveValuesPerTraining)
    f1 = mean(f1PerTraining)
    f1Std = std(f1PerTraining)
    matrix = mean(confusionMatrixPerTraining, dims=3)

  end


  return (acc, accStd, errorRate, errorRateStd, sensibility, sensibilityStd, specificity, specificityStd,
    precision, precisionStd, negativePredictiveValues, negativePredictiveValuesStd, f1, f1Std, matrix)

end


function set_modelHyperparameters(; kernel::String="linear", C::Float64=0.0,
  degree::Int64=0, gamma::Float64=0.0,
  coef0::Float64=0.0, topology::Array{Int64,1}=[2, 2],
  learningRate::Float64=0.01, validationRatio::Float64=0.2,
  numExecutions::Int64=50, maxEpochs::Int64=1000,
  maxEpochsVal::Int64=6, transferFunctions::Array{Function,1}=[Flux.relu, Flux.sigmoid],
  n_neighbors::Int64=5, max_depth::Int64=6, minLoss::Real=0.0)


  dict = Dict{String,Any}()

  # Params for ANN
  dict["topology"] = topology
  dict["learningRate"] = learningRate
  dict["validationRatio"] = validationRatio
  dict["maxEpochs"] = maxEpochs
  dict["maxEpochsVal"] = maxEpochsVal
  dict["transferFunctions"] = transferFunctions
  dict["minLoss"] = minLoss
  @assert validationRatio < 1.0 "The value of validationRatio must be less than 1"

  # Params for SVC, DecissionTreeClassifier and KNeighborsClassifier
  dict["kernel"] = kernel
  dict["C"] = C
  dict["degree"] = degree
  dict["gamma"] = gamma
  dict["coef0"] = coef0

  dict["max_depth"] = max_depth

  dict["n_neighbors"] = n_neighbors

  # Common params
  dict["numExecutions"] = numExecutions

  return dict
end

end
