module CrossValidation


export crossvalidation, ANNCrossValidation

    include("boletin02.jl")
    include("boletin03.jl")
    import .ANNUtils: oneHotEncoding, trainClassANN
    import .Overtraining: holdOut

    using Random;
    using Flux;


    function crossvalidation(N::Int64, k::Int)
        auxVector = 1:k;

        repeats = Int64(ceil(N/k));
        fullVector = repeat(auxVector, outer = repeats);
        fullVector = fullVector[1:N];

        shuffle!(fullVector);
        
        
        return fullVector;
    end

    function crossvalidation(targets::AbstractArray{Bool, 1}, k::Int)
        indexes = zeros(Int64, 1, size(targets, 1));

        posInstances = findall(targets);
        posGroups = crossvalidation(size(posInstances, 1), k);
        indexes[posInstances] .= posGroups;

        negInstances = findall(.~targets);
        negGroups = crossvalidation(size(negInstances, 1), k);
        indexes[negInstances] .= negGroups;

        return indexes;
    end


    function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
        indexes = zeros(Int64, size(targets, 1))

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
        
        precisiones = zeros(1, numClassifiers);
        tasasError = zeros(1, numClassifiers);
        sensibilidades = zeros(1, numClassifiers);
        especificidades = zeros(1, numClassifiers);
        VPPs = zeros(1, numClassifiers);
        VPNs = zeros(1, numClassifiers);
        F1s = zeros(1, numClassifiers);
        
        targets = oneHotEncoding(targets);

        for fold in 1:numFolds
            iterIndexes = findall(crossValidationIndices .== fold)
            if(validationRatio == 0)
                trainInputs = inputs[iterIndexes]
                trainOutputs = targets[iterIndexes]
            else
                #holdOut para separar entrenamiento y validación
            end
            #testInputs = 
            #testOutputs = 
            #como nosotros los tenemos separados no sé cómo pasárselos

            precisionesFold = zeros(1, numClassifiers);
            tasasErrorFold = zeros(1, numClassifiers);
            sensibilidadesFold = zeros(1, numClassifiers);
            especificidadesFold = zeros(1, numClassifiers);
            VPPsFold = zeros(1, numClassifiers);
            VPNsFold = zeros(1, numClassifiers);
            F1sFold = zeros(1, numClassifiers);

            for exec in 1:numExecutions
                #llamar a trainClassAnn y evaluarla con confusionMatrix
            end
            #hacer la media de los resultados obtenidos en confusionMatrix
        end
    
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

sol = crossvalidation(bools, 4);
println(typeof(sol))
print(sol);
