module ANNUtilsRegression

using Statistics
using Flux
using Flux.Losses

export oneHotEncoding, calculateMinMaxNormalizationParameters, calculateZeroMeanNormalizationParameters,
    normalizeMinMax!, normalizeMinMax, normalizeZeroMean!, normalizeZeroMean,
    classifyOutputs, accuracy, buildClassANN, trainClassANN

# ONE HOT ENCODING
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    num_patterns = length(feature)

    if num_classes == 2
        encoded_matrix = vec(reshape(feature .== classes[1], :, 1))
    else
        encoded_matrix = falses(num_patterns, num_classes)

        for i in 1:num_classes
            encoded_matrix[:, i] .= (==).(feature, classes[i])
        end
    end

    return encoded_matrix
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) =
    oneHotEncoding(feature, unique(feature))

oneHotEncoding(feature::AbstractArray{Bool,1}) =
    reshape(feature, size(feature, 1), 1)

# CALCULATE NORMALIZATION PARAMETERS
## MIN AND MAX
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (minimum(dataset, dims=1), maximum(dataset, dims=1))
end

## ZERO MEAN AND STD
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (Statistics.mean(dataset, dims=1), Statistics.std(dataset, dims=1))
end

# NORMALIZE
## MIN AND MAX
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    dataset .-= normalizationParameters[1]
    dataset ./= normalizationParameters[2]
end


function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    min_value, max_value = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, (min_value, max_value))
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    return (dataset .- normalizationParameters[1]) ./ normalizationParameters[2]
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    min_value, max_value = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax(dataset, (min_value, max_value))
end

## ZERO MEAN AND STD
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    dataset .-= normalizationParameters[1]
    dataset ./= normalizationParameters[2]
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    mean, std = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, (mean, std))
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    return (dataset .- normalizationParameters[1]) ./ normalizationParameters[2]
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    mean, std = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, (mean, std))
end

# CLASIFY OUTPUTS
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .> threshold
end


# Porque hay dos funciones con el mismo nombre y los mismos argumentos?

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    num_columns = size(outputs, 2)

    if num_columns == 1
        return classifyOutputs(outputs[:, 1], threshold=threshold)
    end

    (_, index) = findmax(outputs, dims=2)
    out = falses(size(outputs))

    return out[index] .= true
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    num_features = size(outputs, 2)

    if num_features == 1
        return classifyOutputs(outputs[:, 1], threshold=threshold)
    end

    (_, index_with_max_position) = findmax(outputs, dims=2)
    out = falses(size(outputs))
    out[index_with_max_position] .= true
    return out
end

# ACCURACY
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end



function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    num_features = size(outputs, 2)
    if num_features == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    end

    classComparison = targets .== outputs
    correctClassifications = all(classComparison, dims=2)
    return mean(correctClassifications)
end



function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return accuracy(classifyOutputs(outputs, threshold=threshold), targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    outputs = outputs .> threshold
    num_features = size(outputs, 2)
    if num_features == 1
        return accuracy(targets[:, 1], outputs[:, 1])
    end

    return accuracy(targets, outputs)
end

# BUILD CLASS ANN
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)))

    ann = Chain()
    numInputsLayer = numInputs
    iterTransFunction = 1

    if length(topology) != 0
        for numOutputsLayer = topology
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[iterTransFunction]))
            numInputsLayer = numOutputsLayer
            iterTransFunction += 1
        end
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, sigmoid))
        return ann
    end

    ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
    ann = Chain(ann..., softmax)

    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    @assert size(train_inputs, 1) == size(train_targets, 1) "Train_inputs and train_targets must have the same number of samples"
    @assert size(val_inputs, 1) == size(val_targets, 1) "Val_inputs and val_targets must have the same number of samples"
    @assert size(test_inputs, 1) == size(test_targets, 1) "Test_inputs and test_targets must have the same number of samples"

    ann = buildClassANN(size(train_inputs, 2), topology, size(train_targets, 2), transferFunctions=transferFunctions)

    loss(x, y) = (size(y, 1) == 1) ? Losses.binary_focal_loss(ann(x), y) : Losses.focal_loss(ann(x), y)

    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]

    trainingLoss = loss(train_inputs', train_targets')
    push!(trainingLosses, trainingLoss)

    if !isempty(val_inputs)
        validationLoss = loss(val_inputs', val_targets')
        testLoss = loss(test_inputs', test_targets')
        push!(validationLosses, validationLoss)
        push!(testLosses, testLoss)
    end

    numEpoch = 0
    numEpochVal = 0

    bestAnn = deepcopy(ann)

    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochVal < maxEpochsVal)

        Flux.train!(loss, Flux.params(ann), [(train_inputs', train_targets')], ADAM(learningRate))

        trainingLoss = loss(train_inputs', train_targets')

        numEpoch += 1

        if !isempty(val_inputs)
            validationLoss = loss(val_inputs', val_targets')
            testLoss = loss(test_inputs', test_targets')
            push!(validationLosses, validationLoss)
            push!(testLosses, testLoss)
            if validationLoss < validationLosses[end-1]
                bestAnn = deepcopy(ann)
                numEpochVal = 0
            else
                numEpochVal += 1
            end

            #println("Epoch ", numEpoch, ": loss: ", trainingLoss, " validation loss: ", validationLoss, " test loss: ", testLoss)
        else
            #println("Epoch ", numEpoch, ": loss: ", trainingLoss)
        end



        push!(trainingLosses, trainingLoss)

    end

    if isempty(validationLosses)
        bestAnn = deepcopy(ann)
    end

    return (bestAnn, trainingLosses, validationLosses, testLosses)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    return trainClassANN(topology, (train_inputs, reshape(train_targets, length(train_targets), 1));
        validationDataset=(val_inputs, reshape(val_targets, length(val_targets), 1)),
        testDataset=(test_inputs, reshape(test_targets, length(test_targets), 1)),
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)

end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    ann = Chain()
    numInputsLayer = numInputs
    iterTransFunction = 1

    if length(topology) != 0
        for numOutputsLayer = topology
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[iterTransFunction]))
            numInputsLayer = numOutputsLayer
            iterTransFunction += 1
        end
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, sigmoid))
        return ann
    end

    ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
    ann = Chain(ann..., softmax)

    return ann
end

function buildRegANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(sigmoid, length(topology)))

    ann = Chain()
    numInputsLayer = numInputs
    iterTransFunction = 1

    if length(topology) != 0
        for numOutputsLayer = topology
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[iterTransFunction]))
            numInputsLayer = numOutputsLayer
            iterTransFunction += 1
        end
    end

    ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))

    return ann
end

function trainRegANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    @assert size(train_inputs, 1) == size(train_targets, 1) "Train_inputs and train_targets must have the same number of samples"
    @assert size(val_inputs, 1) == size(val_targets, 1) "Val_inputs and val_targets must have the same number of samples"
    @assert size(test_inputs, 1) == size(test_targets, 1) "Test_inputs and test_targets must have the same number of samples"

    ann = buildClassANN(size(train_inputs, 2), topology, 1, transferFunctions=transferFunctions)

    loss(x, y) = Losses.mse(ann(x), y)

    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]

    trainingLoss = loss(train_inputs', train_targets')
    push!(trainingLosses, trainingLoss)

    if !isempty(val_inputs)
        validationLoss = loss(val_inputs', val_targets')
        testLoss = loss(test_inputs', test_targets')
        push!(validationLosses, validationLoss)
        push!(testLosses, testLoss)
    end

    numEpoch = 0
    numEpochVal = 0

    bestAnn = deepcopy(ann)

    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochVal < maxEpochsVal)

        Flux.train!(loss, Flux.params(ann), [(train_inputs', train_targets')], ADAM(learningRate))

        trainingLoss = loss(train_inputs', train_targets')

        numEpoch += 1

        if !isempty(val_inputs)
            validationLoss = loss(val_inputs', val_targets')
            testLoss = loss(test_inputs', test_targets')
            push!(validationLosses, validationLoss)
            push!(testLosses, testLoss)
            if validationLoss < validationLosses[end-1]
                bestAnn = deepcopy(ann)
                numEpochVal = 0
            else
                numEpochVal += 1
            end

            #println("Epoch ", numEpoch, ": loss: ", trainingLoss, " validation loss: ", validationLoss, " test loss: ", testLoss)
        else
            #println("Epoch ", numEpoch, ": loss: ", trainingLoss)
        end



        push!(trainingLosses, trainingLoss)

    end

    if isempty(validationLosses)
        bestAnn = deepcopy(ann)
    end

    return (bestAnn, trainingLosses, validationLosses, testLosses)

end

function trainRegANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    (train_inputs, train_targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    print(size(train_targets))
    @assert size(train_targets, 2) == 1 "train_targets must ve a 1 dimensional array"

    return trainRegANN(topology, (train_inputs, vec(train_targets)), validationDataset=(val_inputs, vec(val_targets)),
     testDataset=(test_inputs, vec(test_targets)), transferFunctions=transferFunctions, maxEpochs=maxEpochs,
     minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)

end

end
