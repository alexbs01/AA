using Random;
using Flux;

# FUNCIONES DE PRÁCTICAS ANTERIORES

function oneHotEncoding(feature::AbstractArray{<:Any, 1}, classes::AbstractArray{:Any, 1})
    numClasses = length(classes);

    @assert (numClasses>1) "Para un problema de clasificación debe haber más de una clase";
    if(numClasses==2)
        targets = reshape(targets.==classes[1], :, 1);
    
    else
        targets = BitArray{2}(undef, length(features), numClasses); #La longitud de features es el número de patrones que hay
        for numClass = 1::numClasses
            targets[:,numClass] .= (features.==classes[numClass]);
        end
    end
    return targets;
end

function oneHotEncoding(feature::AbstractArray{<:Any, 1})
    classes = unique(feature);
    return oneHotEncoding(feature, classes);
end


function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1);
end


#FUNCIONES NUEVAS

function crossvalidation(N::Int64, k::Int)
    auxVector = 1:k;

    repeats = Int64(ceil(N/k));
    fullVector = repeat(auxVector, outer = repeats);
    fullVector = fullVector[1:N];

    shuffle!(fullVector);
    return fullVector;
end

function crossvalidation(targets::AbstractArray{Bool, 1}, k::Int)
    indexes = zeros(1, size(targets, 1));

    posInstances = findall(targets);
    posGroups = crossvalidation(size(posInstances, 1), k);
    indexes[posInstances] .= posGroups;

    negInstances = findall(.~targets);
    negGroups = crossvalidation(size(negInstances, 1), k);
    indexes[negInstances] .= negGroups;

    return indexes;
end


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
    indexes = zeros(1, size(targets, 1));
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

    if(length(classes) == 2)
        return crossvalidation(targets==classes[1], k);
    else
        
        return crossvalidation(oneHotEncoding(targets));
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

    for fold = i:numExecutions
        
    end

end

#PRUEBAS


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

print(sol);
