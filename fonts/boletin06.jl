module ScikitModels

include("boletin02.jl")
include("boletin03.jl")
include("boletin04.jl")
include("../crossValWithRegression.jl")
include("boletin05.jl")
include("../errorFunctions/errorFunctions.jl")
include("../annWithRegression.jl")

using ScikitLearn
using ScikitLearn: fit!, predict
using Flux
using Flux.Losses
using Statistics
using .Metrics: confusionMatrix
using .ANNUtilsRegression: oneHotEncoding, trainRegANN
using .Overtraining: holdOut
using .RegCrossValidation: regANNCrossValidation, ANNCrossValidation
using .ErrorFunctions: errorFunction

export modelCrossValidation, set_modelHyperparameters

@sk_import svm:SVC
@sk_import svm:SVR
@sk_import tree:DecisionTreeClassifier
@sk_import tree:DecisionTreeRegressor
@sk_import neighbors:KNeighborsClassifier
@sk_import neighbors:KNeighborsRegressor

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
  inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
  crossValidationIndices::Array{Int64,1}, doRegression::Bool=false)

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


    if doRegression
      (mse, mseStd), (mae, maeStd), (msle, msleStd), (rmse, rmseStd) =
        regANNCrossValidation(topology, inputs, targets, crossValidationIndices, numExecutions=numExecutions,
          transferFunctions=transferFunctions, maxEpochs=maxEpochs, learningRate=learningRate,
          validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)

      return (mse, mseStd, mae, maeStd, msle, msleStd, rmse, rmseStd)
    end

    (acc, accStd), (errorRate, errorRateStd), (sensibility, sensibilityStd), (specificity, specificityStd),
    (precision, precisionStd), (negativePredictiveValues, negativePredictiveValuesStd), (f1, f1Std), matrix =
      ANNCrossValidation(topology, inputs, targets, crossValidationIndices, numExecutions=numExecutions,
        transferFunctions=transferFunctions, maxEpochs=maxEpochs, learningRate=learningRate,
        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)

  else

    numFolds = maximum(crossValidationIndices)
    numClasses = length(unique(targets))

    accuracyPerTraining = zeros(numFolds)
    errorRatePerTraining = zeros(numFolds)
    sensibilityPerTraining = zeros(numFolds)
    specificityPerTraining = zeros(numFolds)
    precisionPerTraining = zeros(numFolds)
    negativePredictiveValuesPerTraining = zeros(numFolds)
    f1PerTraining = zeros(numFolds)
    confusionMatrixPerTraining = zeros(numClasses, numClasses, numFolds)

    mse = zeros(numFolds)
    mae = zeros(numFolds)
    msle = zeros(numFolds)
    rmse = zeros(numFolds)

    for numFold in 1:numFolds

      if modelType == :SVC
        possibleKernel = ["linear", "poly", "rbf", "sigmoid"]
        C = modelHyperparameters["C"]
        kernel = modelHyperparameters["kernel"]
        degree = modelHyperparameters["degree"]
        gamma = modelHyperparameters["gamma"]
        coef0 = modelHyperparameters["coef0"]
        @assert kernel in possibleKernel "Kernel must be linear, poly, rbf or sigmoid"
        classWeight = "balanced"

        if kernel == "linear"
          if !doRegression
            model = SVC(kernel=kernel, C=C, class_weight=classWeight)
          else
            model = SVR(kernel=kernel, C=C)
          end
        elseif kernel == "poly"
          if !doRegression
            model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0, class_weight=classWeight)
          else
            model = SVR(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0)
          end
        elseif kernel == "rbf"
          if !doRegression
            model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=classWeight)
          else
            model = SVR(kernel=kernel, C=C, gamma=gamma)
          end
        elseif kernel == "sigmoid"
          if !doRegression
            model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0, class_weight=classWeight)
          else
            model = SVR(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
          end
        end

      elseif modelType == :DecissionTreeClassifier
        max_depth = modelHyperparameters["max_depth"]

        if !doRegression
          model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
        else
          model = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        end


      elseif modelType == :KNeighborsClassifier
        n_neighbors = modelHyperparameters["n_neighbors"]

        if !doRegression
          model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else
          model = KNeighborsRegressor(n_neighbors=n_neighbors)
        end
      end

      trainIndex = findall(crossValidationIndices .!= numFold)
      testIndex = findall(crossValidationIndices .== numFold)

      trainingInputs = inputs[trainIndex, :]
      trainingTargets = targets[trainIndex]
      testingInputs = inputs[testIndex, :]
      testingTargets = targets[testIndex]

      model = fit!(model, trainingInputs, trainingTargets)

      outputs = predict(model, testingInputs)

      if !doRegression
        (
          accuracyPerTraining[numFold],
          errorRatePerTraining[numFold],
          sensibilityPerTraining[numFold],
          specificityPerTraining[numFold],
          precisionPerTraining[numFold],
          negativePredictiveValuesPerTraining[numFold],
          f1PerTraining[numFold],
          confusionMatrixPerTraining[:, :, numFold],
        ) = confusionMatrix(outputs, testingTargets)
      else
        (mse[numFold], mae[numFold], msle[numFold], rmse[numFold]) = errorFunction(Float32.(testingTargets), outputs)

        (
          accuracyPerTraining[numFold],
          errorRatePerTraining[numFold],
          sensibilityPerTraining[numFold],
          specificityPerTraining[numFold],
          precisionPerTraining[numFold],
          negativePredictiveValuesPerTraining[numFold],
          f1PerTraining[numFold],
          confusionMatrixPerTraining[:, :, numFold],
        ) = confusionMatrix(encoder(outputs, sort(unique(testingTargets))), testingTargets)

      end

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

    if (doRegression)
      msem = mean(mse)
      mseDes = std(mse)
      maem = mean(mae)
      maeDes = std(mae)
      mslem = mean(msle)
      msleDes = std(msle)
      rmsem = mean(rmse)
      rmseDes = std(rmse)

      return (acc, accStd, errorRate, errorRateStd, sensibility, sensibilityStd,
        specificity, specificityStd, precision, precisionStd, negativePredictiveValues,
        negativePredictiveValuesStd, f1, f1Std, matrix, msem, mseDes, maem, maeDes,
        mslem, msleDes, rmsem, rmseDes)
    end
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

function encoder(vector::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
  numClasses = length(classes)
  arrMids = []

  for i in 2:numClasses
    push!(arrMids, ((classes[i] - classes[i-1]) / 2) + classes[i-1])
  end

  encoded = falses(length(vector), numClasses)

  encoded[:, 1] .= (<=).(vector, arrMids[1])
  for i in 2:(numClasses-1)
    encoded[:, i] .= (==).(((<=).(vector, arrMids[i])), ((>).(vector, arrMids[i-1])))
  end
  encoded[:, numClasses] .= (>).(vector, arrMids[numClasses-1])

  return encoded
end

end
