using JLD2
using Base: setindex, indexed_iterate
using FileIO
using DelimitedFiles
using Statistics
using Random
using Flux
using Flux.Losses

# oneHotEncoding
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
  numClasses = length(classes)
  if numClasses == 2
    feature = reshape(feature .== classes[1], :, 1)
  else
    aux = Array{Bool,2}(undef, length(feature), numClasses)
    for class in 1:numClasses
      aux[:, class] .= (feature .== classes[class])
    end
    feature = aux
  end
  return feature'
end

function oneHotEncoding(feature::AbstractArray{<:Any,1})
  return oneHotEncoding(feature, unique(feature))
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
  return oneHotEncoding(feature, unique(feature))
end

# calculateMinMaxNormalizationParameters

function calculateMinMaxNormalizationParameters(input::AbstractArray{<:Real,2})
  max = maximum(input, dims=1)
  min = minimum(input, dims=1)
  return (min, max)
end

# calculateZeroMeanNormalizationParameters

function calculateZeroMeanNormalizationParameters(input::AbstractArray{<:Real,2})
  avg = mean(input, dims=1)
  svg = std(input, dims=1)
  return (avg, svg)
end

# normalizeMinMax!

function normalizeMinMax!(input::AbstractArray{<:Real,2}, MinMax::NTuple{2,AbstractArray{<:Real,2}})
  input .-= MinMax[1]
  input ./= (MinMax[2] .- MinMax[1])
  input[:, vec(MinMax[1] .== MinMax[2])] .= 0
end

function normalizeMinMax!(input::AbstractArray{<:Real,2})
  normalizeMinMax!(input, calculateMinMaxNormalizationParameters(input))
end

# normalizeMinMax

function normalizeMinMax(input::AbstractArray{<:Real,2}, MinMax::NTuple{2,AbstractArray{<:Real,2}})
  inputAux = copy(input)
  normalizeMinMax!(inputAux, MinMax)
  return inputAux
end

function normalizeMinMax(input::AbstractArray{<:Real,2})
  return normalizeMinMax(input, calculateMinMaxNormalizationParameters(input))
end

# normalizeZeroMean!

function normalizeZeroMean!(input::AbstractArray{<:Real,2}, ZeroMean::NTuple{2,AbstractArray{<:Real,2}})
  input .-= ZeroMean[1]
  input ./= ZeroMean[2]
  input[:, vec(ZeroMean[2] .== 0)] .= 0
end

function normalizeZeroMean!(input::AbstractArray{<:Real,2})
  normalizeZeroMean!(input, calculateZeroMeanNormalizationParameters(input))
end

# normalizeZeroMean

function normalizeZeroMean(input::AbstractArray{<:Real,2}, ZeroMean::NTuple{2,AbstractArray{<:Real,2}})
  inputAux = copy(input)
  normalizeZeroMean!(inputAux, ZeroMean)
  return inputAux
end

function normalizeZeroMean(input::AbstractArray{<:Real,2})
  return normalizeZeroMean(input, calculateZeroMeanNormalizationParameters(input))
end

# classifyOutputs

function classifyOutputs(outputs::AbstractArray{<:Real,1}, threshold::Real=0.5)
  return broadcast(>=, outputs, threshold)
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold::Real=0.5)
  if size(outputs, 2) == 1
    return reshape(classifyOutputs(outputs[:], threshold), :, 1)
  else
    (_, maxIndex) = findmax(outputs, dims=2)
    outputs = falses(size(outputs))
    outputs[maxIndex] .= true
    return outputs
  end

end

# accuracy

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
  compared = targets .== outputs
  return mean(compared)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
  if size(outputs, 2) == 1
    return accuracy(outputs[:], targets[:])
  else
    compared = targets .== outputs
    correct = all(compared, dims=2)
    return mean(correct)
  end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, threshold::Real=0.5)
  return accuracy(classifyOutputs(outputs, threshold), targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, threshold::Real=0.5)
  return accuracy(classifyOutputs(outputs, threshold), targets)
end

# buildClassAnn

function buildClassANN(numInputs::Int,
  topology::AbstractArray{<:Int,1},
  numOutputs::Int,
  transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

  ann = Chain()
  numInputsLayer = numInputs

  if size(topology, 1) != 0
    for i in eachindex(topology)
      numOutputsLayer = topology[i]
      ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))
      numInputsLayer = numOutputsLayer
    end
  end

  if numOutputs > 1
    ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
    ann = Chain(ann..., softmax)
  else
    ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
  end

  return ann
end

# holdOut

function holdOut(N::Int, P::Real)
  index = randperm(N)
  NF = round(Int, Real(N) * P)
  return (index[1:(N-NF)], index[(N+1-NF):end])
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
  index = randperm(N)
  NV = round(Int, Real(N) * Pval)
  NT = round(Int, Real(N) * Ptest)
  return (index[1:(N-NV-NT)], index[(N-NV-NT+1):(N-NT)], index[(N-NT+1):end])
end

# trainClassANN v2

function trainClassANN(topology::AbstractArray{<:Int,1},
  trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
  validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
  testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0, 0)),
  transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
  maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
  maxEpochsVal::Int=20)

  ann = buildClassANN(size(trainingDataset[1], 2), topology, size(trainingDataset[2], 2), transferFunctions)
  loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
  opt_state = Flux.setup(Adam(learningRate), ann)

  lossTrain = loss(ann, trainingDataset[1]', trainingDataset[2]')
  lossTrainArr = [lossTrain]
  lossVal = loss(ann, validationDataset[1]', validationDataset[2]')
  lossValArr = [lossVal]
  lossTest = loss(ann, testDataset[1]', testDataset[2]')
  lossTestArr = [lossTest]


  BestAnn = deepcopy(ann)
  BestVal = lossVal

  contCiles = maxEpochs
  contVal = 0

  while lossTrain > minLoss && contCiles > 0 && contVal <= maxEpochsVal
    Flux.train!(loss, ann, [(trainingDataset[1]', trainingDataset[2]')], opt_state)

    lossTrain = loss(ann, trainingDataset[1]', trainingDataset[2]')
    push!(lossTrainArr, copy(lossTrain))

    lossVal = loss(ann, validationDataset[1]', validationDataset[2]')
    push!(lossValArr, copy(lossVal))

    if BestVal >= lossVal
      BestAnn = deepcopy(ann)
      BestVal = copy(lossVal)
      contVal = 0
    else
      contVal = contVal + 1
    end

    lossTest = loss(ann, testDataset[1]', testDataset[2]')
    push!(lossTestArr, copy(lossTest))

    contCiles = contCiles - 1
  end

  return (BestAnn, lossTrainArr, lossValArr, lossTestArr)
end

function trainClassANN(topology::AbstractArray{<:Int,1},
  trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}},
  validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
  (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
  testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
  (Array{eltype(trainingDataset[1]),2}(undef, 0, 0), falses(0)),
  transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
  maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
  maxEpochsVal::Int=20)

  return trainClassANN(topology,
    (trainingDataset[1], reshape(trainingDataset[2], :, 1)),
    (validationDataset[1], reshape(validationDataset[2], :, 1)),
    (testDataset[1], reshape(testDataset[2], :, 1)),
    transferFunctions,
    maxEpochs, minLoss, learningRate,
    maxEpochsVal)
end


data = load_object("VH-VL.jld2")


index = holdOut(size(data, 1), 0.2, 0.2)


Normalization = calculateMinMaxNormalizationParameters(Float32.(data[:, 1:6]))

inputTr = data[index[1], 1:6]
inputVl = data[index[2], 1:6]
inputTs = data[index[3], 1:6]
inputTr = Float32.(inputTr)
inputVl = Float32.(inputVl)
inputTs = Float32.(inputTs)

targtTr = data[index[1], 7]
targtVl = data[index[2], 7]
targtTs = data[index[3], 7]
targtTr = oneHotEncoding(targtTr)
targtVl = oneHotEncoding(targtVl)
targtTs = oneHotEncoding(targtTs)

topology = [7, 5, 4]


ann = trainClassANN(topology,
  (normalizeMinMax(inputTr, Normalization), targtTr'),
  (normalizeMinMax(inputVl, Normalization), targtVl'),
  (normalizeMinMax(inputTs, Normalization), targtTs'))


save_object("annAndLoss.jld2", ann)

output = ann[1](normalizeMinMax(inputTs, Normalization)')
println(accuracy(output', targtTs'))

