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

        @assert (numElems > k) "El número de elementos de cada clase debe ser mayor que k";

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
        return crossvalidation(targets==classes[1], k);
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
    
    precisiones = zeros(2, numClassifiers, numFolds);
    tasasError = zeros(2, numClassifiers, numFolds);
    sensibilidades = zeros(2, numClassifiers, numFolds);
    especificidades = zeros(2, numClassifiers, numFolds);
    VPPs = zeros(2, numClassifiers, numFolds);
    VPNs = zeros(2, numClassifiers, numFolds);
    F1s = zeros(2, numClassifiers, numFolds);
    
    targets = oneHotEncoding(targets);

    for fold in 1:numFolds
        testIndexes = findall(crossValidationIndices .== fold)  #Para cada fold, se usan los elementos correspondientes como conjunto de test
        trainIndexes = findall(crossValidationIndices .!= fold) #El resto será el conjunto de entrenamiento
        
        trainInputs = inputs[trainIndexes]
        trainTargets = targets[trainIndexes]
        testInputs = inputs[testIndexes]
        testTargets =  targets[testIndexes]

        if(validationRatio != 0)
            trainSize = length(trainInputs);
            validationRatio = validationRatio * trainSize/(length(crossValidationIndices))
            trainIndexes, valIndexes = holdOut(trainSize, validationRatio)

            trainTargets = trainTargets[trainIndexes]
            validationTargets = validationTargets[trainIndexes]
        end

        precisionesFold = zeros(2, numClassifiers, numExecutions);
        tasasErrorFold = zeros(2, numClassifiers, numExecutions);
        sensibilidadesFold = zeros(2, numClassifiers, numExecutions);
        especificidadesFold = zeros(2, numClassifiers, numExecutions);
        VPPsFold = zeros(2, numClassifiers, numExecutions);
        VPNsFold = zeros(2, numClassifiers, numExecutions);
        F1sFold = zeros(2, numClassifiers, numExecutions);

        for exec in 1:numExecutions
            #llamar a trainClassAnn y evaluarla con confusionMatrix
            (bestAnn, trainingLosses, validationLosses, testLosses) = trainClassANN(topology, (trainInputs, trainOutputs), (validationInputs, validationOutputs), (testInputs, testOutputs), 
            transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal)
            outputs = bestAnn(testInputs)

            (precisionesFold[exec], tasasErrorFold[exec], sensibilidadesFold[exec], especificidadesFold[exec], VPPsFold[exec]
            VPNsFold[exec], F1sFold[exec],)confusionMatrix(outputs, targets)
        end
        #hacer la media de los resultados obtenidos en confusionMatrix
        precisiones[fold] = mean(precisionesFold)
        tasasError[fold] = mean(tasasErrorFold)
        sensibilidades[fold] = mean(sensibilidadesFold)
        especificidades[fold] = mean(especificidadesFold)
        VPPs[fold] = mean(VPPsFold)
        VPNs[fold] = mean(VPNsFold)
        F1s[fold] = mean(F1sFold)
    end
    return (mean(precisiones, std(precisiones))), (mean(tasasError, std(tasasError))), (mean(sensibilidades), std(sensibilidades)),
     (mean(especificidades), std(especificidades)), (mean(VPPs), std(VPPs)), (mean(VPNs), std(VPNs)), (mean(F1s), std(F1s))
    end

end

#PRUEBAS

import .CrossValidation: crossvalidation

bools = [
    true false false false;
    false false true false;
    false true false false;
    true false false false;
    false false false true;
    false true false false;
    false false false true;
    false false true false;
    true false false false;
    false false true false;
    false true false false;
    true false false false;
    false false false true;
    false true false false;
    false false false true;
    false false true false;
    true false false false;
    false false true false;
    false true false false;
    true false false false;
    false false false true;
    false true false false;
    false false false true;
    false false true false
]

sol = (crossvalidation(bools, 4));
println(typeof(sol))
print(sol);
