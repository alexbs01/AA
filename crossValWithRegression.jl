module RegCrossValidation


export crossvalidation, ANNCrossValidation, regANNCrossValidation

include("fonts/boletin02.jl")
include("fonts/boletin03.jl")
include("fonts/boletin04.jl")
include("annWithRegression.jl")
include("errorFunctions/errorFunctions.jl")

import .ANNUtilsRegression: oneHotEncoding, trainRegANN, trainClassANN
import .Overtraining: holdOut
import .Metrics: confusionMatrix
import .ErrorFunctions: errorFunction

using Random
using Flux
using Statistics


function crossvalidation(N::Int64, k::Int)
    auxVector = 1:k

    repeats = Int64(ceil(N / k))
    fullVector = repeat(auxVector, outer=repeats)
    fullVector = fullVector[1:N]

    shuffle!(fullVector)


    return fullVector
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int)
    indexes = zeros(Int64, size(targets, 1))

    posInstances = findall(targets)
    posGroups = crossvalidation(size(posInstances, 1), k)
    indexes[posInstances] .= posGroups

    negInstances = findall(.~targets)
    negGroups = crossvalidation(size(negInstances, 1), k)
    indexes[negInstances] .= negGroups

    return indexes
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
    indexes = zeros(Int64, size(targets, 1))
    for numClass = 1:size(targets, 2)
        class = targets[:, numClass]
        numElems = sum(class)

        @assert (numElems >= k) "El número de elementos de cada clase debe ser mayor que k"

        groups = crossvalidation(numElems, k)

        instances = findall(class)
        indexes[instances] .= groups
    end

    return indexes
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int)
    classes = unique(targets)

    @assert (length(classes) > 1) "Es necesario un mínimo de dos clases"

    if length(classes) == 2
        return crossvalidation((targets .== classes[1]), k)
    else

        return crossvalidation(oneHotEncoding(targets), k)
    end
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1}; numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0,
    learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20, classification::Bool=true)

    numFolds = maximum(crossValidationIndices)

    numClassifiers = length(unique(targets))


    targets = oneHotEncoding(targets)

    accuracy = zeros(numFolds)
    errorRate = zeros(numFolds)
    recall = zeros(numFolds)
    specificity = zeros(numFolds)
    precision = zeros(numFolds)
    negativePredictiveValue = zeros(numFolds)
    F1s = zeros(numFolds)
    matrixes = zeros(numClassifiers, numClassifiers, numExecutions)

    for fold in 1:numFolds
        testIndexes = findall(crossValidationIndices .== fold)  #Para cada fold, se usan los elementos correspondientes como conjunto de test
        trainIndexes = findall(crossValidationIndices .!= fold) #El resto será el conjunto de entrenamiento

        trainInputs = inputs[trainIndexes, :]
        testInputs = inputs[testIndexes, :]

        trainTargets = targets[trainIndexes, :]
        testTargets = targets[testIndexes, :]


        foldAccuracy = zeros(numExecutions)
        foldErrorRate = zeros(numExecutions)
        foldRecall = zeros(numExecutions)
        foldSpecificity = zeros(numExecutions)
        foldPrecision = zeros(numExecutions)
        foldNPV = zeros(numExecutions)
        foldF1s = zeros(numExecutions)
        foldMatrix = zeros(numClassifiers, numClassifiers, numExecutions)

        trainSize = length(trainIndexes)

        for exec in 1:numExecutions

            #Si validationratio es mayor que 0, 
            if (validationRatio > 0)
                validationRatio2 = validationRatio * numFolds/(numFolds - 1)

                valIndexes, trainIndexes = holdOut(trainSize, validationRatio2)

                validationInputs = trainInputs[valIndexes, :]
                trainInputs2 = trainInputs[trainIndexes, :]

                validationTargets = trainTargets[valIndexes, :]
                trainTargets2 = trainTargets[trainIndexes, :]

                (bestAnn, _, _, _) = trainClassANN(topology, (trainInputs2, trainTargets2),
                    validationDataset=(validationInputs, validationTargets),
                    testDataset=(testInputs, testTargets), learningRate=learningRate, maxEpochs=maxEpochs,
                    maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions, minLoss=minLoss)

            else
                #llamar a trainClassAnn y evaluarla con confusionMatrix
                (bestAnn, _, _, _) = trainClassANN(topology, (trainInputs, trainTargets),
                    testDataset=(testInputs, testTargets), learningRate=learningRate, maxEpochs=maxEpochs,
                    maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions, minLoss=minLoss)
            end

            outputs = collect(bestAnn(testInputs')')

            if numClassifiers == 2
                outputs = vec(outputs)
                testTargets2 = vec(testTargets)
            else
                testTargets2 = testTargets
            end

            (foldAccuracy[exec], foldErrorRate[exec], foldRecall[exec], foldSpecificity[exec], foldPrecision[exec],
                foldNPV[exec], foldF1s[exec], foldMatrix[:, :, exec]) = confusionMatrix(outputs, testTargets2)

        end
        #hacer la media de los resultados obtenidos en confusionMatrix
        accuracy[fold] = mean(foldAccuracy)
        errorRate[fold] = mean(foldErrorRate)
        recall[fold] = mean(foldRecall)
        specificity[fold] = mean(foldSpecificity)
        precision[fold] = mean(foldPrecision)
        negativePredictiveValue[fold] = mean(foldNPV)
        F1s[fold] = mean(foldF1s)
        matrixes[:, :, fold] = mean(foldMatrix, dims=3)

    end

    return (mean(accuracy, dims=1), std(accuracy, dims=1)), (mean(errorRate, dims=1), std(errorRate, dims=1)),
    (mean(recall, dims=1), std(recall, dims=1)), (mean(specificity, dims=1), std(specificity, dims=1)),
    (mean(precision, dims=1), std(precision, dims=1)), (mean(negativePredictiveValue, dims=1),
        std(negativePredictiveValue, dims=1)), (mean(F1s, dims=1), std(F1s, dims=1)), mean(matrixes, dims=3)
end

function regANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1}; numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0,
    learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)

    numFolds = maximum(crossValidationIndices)

    numClassifiers = length(unique(targets))

    targets = Float32.(targets)

    mse = zeros(numFolds)
    mae = zeros(numFolds)
    msle = zeros(numFolds)
    rmse = zeros(numFolds)


    for fold in 1:numFolds
        testIndexes = findall(crossValidationIndices .== fold)  #Para cada fold, se usan los elementos correspondientes como conjunto de test
        trainIndexes = findall(crossValidationIndices .!= fold) #El resto será el conjunto de entrenamiento

        trainInputs = inputs[trainIndexes, :]
        testInputs = inputs[testIndexes, :]

        trainTargets = targets[trainIndexes]
        testTargets = targets[testIndexes]

        
        foldMse = zeros(numExecutions)
        foldMae = zeros(numExecutions)
        foldMsle = zeros(numExecutions)
        foldRmse = zeros(numExecutions)

        for exec in 1:numExecutions

            #Si validationratio es mayor que 0, 
            if (validationRatio > 0)
                validationRatio2 = validationRatio * numFolds/(numFolds - 1)

                valIndexes, trainIndexes = holdOut(trainSize, validationRatio2)

                validationInputs = trainInputs[valIndexes, :]
                trainInputs2 = trainInputs[trainIndexes, :]

                validationTargets = trainTargets[valIndexes, :]
                trainTargets2 = trainTargets[trainIndexes, :]

                (bestAnn, _, _, _) = trainRegANN(topology, (trainInputs2, trainTargets2),
                    validationDataset=(validationInputs, validationTargets),
                    testDataset=(testInputs, testTargets), learningRate=learningRate, maxEpochs=maxEpochs,
                    maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions, minLoss=minLoss)

            else
                #llamar a trainClassAnn y evaluarla con confusionMatrix
                (bestAnn, _, _, _) = trainRegANN(topology, (trainInputs, trainTargets),
                    testDataset=(testInputs, testTargets), learningRate=learningRate, maxEpochs=maxEpochs,
                    maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions, minLoss=minLoss)
            end

            outputs = collect(bestAnn(testInputs')')

            (foldMse[exec], foldMae[exec], foldMsle[exec], foldRmse[exec]) = errorFunction(testTargets, vec(outputs))

        end
        #hacer la media de los resultados obtenidos en confusionMatrix
        mse[fold] = mean(foldMse)
        mae[fold] = mean(foldMae)
        rmse[fold] = mean(foldRmse)
        msle[fold] = mean(foldMsle)


    end

    return (mean(mse, dims=1), std(mse, dims=1)), (mean(mae, dims=1), std(mae, dims=1)),
    (mean(msle, dims=1), std(msle, dims=1)), (mean(rmse, dims=1), std(rmse, dims=1))
end

end