module CrossValidation


export crossvalidation, ANNCrossValidation

include("boletin02.jl")
include("boletin03.jl")
include("boletin04.jl")
import .ANNUtils: oneHotEncoding, trainClassANN
import .Overtraining: holdOut
import .Metrics: confusionMatrix

using Random;
using Flux;
using Statistics


function crossvalidation(N::Int64, k::Int)
    auxVector = 1:k;

    repeats = Int64(ceil(N/k));
    fullVector = repeat(auxVector, outer = repeats);
    fullVector = fullVector[1:N];

    shuffle!(fullVector);


    return fullVector;
end

function crossvalidation(targets::AbstractArray{Bool, 1}, k::Int)
    indexes = zeros(Int64, size(targets, 1));

    posInstances = findall(targets);
    posGroups = crossvalidation(size(posInstances, 1), k);
    indexes[posInstances] .= posGroups;

    negInstances = findall(.~targets);
    negGroups = crossvalidation(size(negInstances, 1), k);
    indexes[negInstances] .= negGroups;

    return indexes;
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
    indexes = zeros(Int64, size(targets, 1));
    for numClass = 1:size(targets, 2)
        class = targets[:, numClass];
        numElems = sum(class);

        @assert (numElems >= k) "El número de elementos de cada clase debe ser mayor que k";

        groups = crossvalidation(numElems, k);

        instances = findall(class);
        indexes[instances] .= groups;
    end

    return indexes;
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int)
    classes = unique(targets);

    @assert (length(classes) > 1) "Es necesario un mínimo de dos clases"

    if length(classes) == 2
        return crossvalidation((targets .== classes[1]), k);
    else    
        
        return crossvalidation(oneHotEncoding(targets), k);
    end
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},crossValidationIndices::Array{Int64,1};numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),maxEpochs::Int=1000, minLoss::Real=0.0,
    learningRate::Real=0.01,validationRatio::Real=0, maxEpochsVal::Int=20)

    numFolds = maximum(crossValidationIndices);
    
    numClassifiers = length(unique(targets));

    if numClassifiers == 2
        numClassifiers = 1
    end
    

    # -------------------REVISAR--------------------------
    precisiones = zeros(numFolds, numClassifiers);
    tasasError = zeros(numFolds, numClassifiers);
    sensibilidades = zeros(numFolds, numClassifiers);
    especificidades = zeros(numFolds, numClassifiers);
    VPPs = zeros(numFolds, numClassifiers);
    VPNs = zeros(numFolds, numClassifiers);
    F1s = zeros(numFolds, numClassifiers);
    #-----------------------------------------------------

    targets = oneHotEncoding(targets);

    for fold in 1:numFolds
        testIndexes = findall(crossValidationIndices .== fold)  #Para cada fold, se usan los elementos correspondientes como conjunto de test
        trainIndexes = findall(crossValidationIndices .!= fold) #El resto será el conjunto de entrenamiento
        
        trainInputs = inputs[trainIndexes, :]
        trainTargets = targets[trainIndexes, :]
        testInputs = inputs[testIndexes, :]
        testTargets =  targets[testIndexes, :]



        precisionesFold = zeros(numExecutions, numClassifiers);
        tasasErrorFold = zeros(numExecutions, numClassifiers);
        sensibilidadesFold = zeros(numExecutions, numClassifiers);
        especificidadesFold = zeros(numExecutions, numClassifiers);
        VPPsFold = zeros(numExecutions, numClassifiers);
        VPNsFold = zeros(numExecutions, numClassifiers);
        F1sFold = zeros(numExecutions, numClassifiers);

        for exec in 1:numExecutions

            #Si validationratio es mayor que 0, 
            if(validationRatio > 0)
                trainSize = length(trainIndexes);
                validationRatio = validationRatio * trainSize/(length(crossValidationIndices))
                trainIndexes, valIndexes = holdOut(trainSize, validationRatio)

                validationInputs = trainInputs[valIndexes, :]
                trainInputs = trainInputs[trainIndexes, :]

                validationTargets = trainTargets[valIndexes, :]
                trainTargets = trainTargets[trainIndexes, :]

                (bestAnn, _, _, _) = trainClassANN(topology, (trainInputs, trainTargets), 
                validationDataset=(validationInputs, validationTargets),
                testDataset=(testInputs, testTargets), learningRate=learningRate, maxEpochs=maxEpochs, 
                maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions, minLoss = minLoss)
            
            else
                #llamar a trainClassAnn y evaluarla con confusionMatrix
                (bestAnn, _, _, _) = trainClassANN(topology, (trainInputs, trainTargets),
                testDataset = (testInputs, testTargets), learningRate=learningRate, maxEpochs=maxEpochs, 
                maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions, minLoss = minLoss)
            end
            outputs = bestAnn(testInputs)

            (precisionesFold[exec], tasasErrorFold[exec], sensibilidadesFold[exec], especificidadesFold[exec], VPPsFold[exec],
            VPNsFold[exec], F1sFold[exec], _) = confusionMatrix(outputs, targets)
        end
        #hacer la media de los resultados obtenidos en confusionMatrix
        precisiones[fold] .= mean(precisionesFold, dims = 1)
        tasasError[fold] .= mean(tasasErrorFold, dims = 1)
        sensibilidades[fold] .= mean(sensibilidadesFold, dims = 1)
        especificidades[fold] .= mean(especificidadesFold, dims = 1)
        VPPs[fold] .= mean(VPPsFold, dims = 1)
        VPNs[fold] .= mean(VPNsFold, dims = 1)
        F1s[fold] .= mean(F1sFold, dims = 1)
    end
    return (mean(precisiones, dims = 1), std(precisiones, dims = 1)), (mean(tasasError, dims = 1), std(tasasError, dims = 1)),
    (mean(sensibilidades, dims = 1), std(sensibilidades, dims = 1)),(mean(especificidades, dims = 1), std(especificidades, dims = 1)),
    (mean(VPPs, dims = 1), std(VPPs, dims = 1)), (mean(VPNs, dims = 1), std(VPNs, dims = 1)), (mean(F1s, dims = 1), std(F1s, dims = 1))
    end

end