using JLD2
using Images
using Statistics
using Random

# Functions that allow the conversion from images to Float64 arrays
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
  matrix = Array{Float64,3}(undef, size(image, 1), size(image, 2), 3)
  matrix[:, :, 1] = convert(Array{Float64,2}, red.(image))
  matrix[:, :, 2] = convert(Array{Float64,2}, green.(image))
  matrix[:, :, 3] = convert(Array{Float64,2}, blue.(image))
  return matrix
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

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

# Functions to load the dataset

VLow = "dataset/train/Very_Low/"
VHigh = "dataset/train/Very_High/"

function meanRGB(image::Array{Float64,3})
  [mean(image[:, :, 1]) mean(image[:, :, 2]) mean(image[:, :, 3])]
end

function stdRGB(image::Array{Float64,3})
  [std(image[:, :, 1]) std(image[:, :, 2]) std(image[:, :, 3])]
end

function imageLoader(folder::String, type::Float64)
  imagArr = Array{Any}(undef, 0, 7)
  for fileName in readdir(folder)
    imag = imageToColorArray(load(string(folder, fileName)))
    imagArr = vcat(imagArr, hcat(meanRGB(imag), stdRGB(imag), type))
  end
  return imagArr
end

VHtr = imageLoader("../dataset/train/Very_High/", 1)
VHva = imageLoader("../dataset/val/Very_High/", 1)
println("VH finished")
VLtr = imageLoader("../dataset/train/Very_Low/", 0)
VLva = imageLoader("../dataset/val/Very_Low/", 0)
println("VL finished")

Htr = imageLoader("../dataset/train/High/", 0.75)
Hva = imageLoader("../dataset/val/High/", 0.75)
println("H finished")
Ltr = imageLoader("../dataset/train/Low/", 0.25)
Lva = imageLoader("../dataset/val/Low/", 0.25)
println("L finished")

data = vcat(VHtr, VLtr)
data = vcat(data, Ltr)
data = vcat(data, Htr)
data = vcat(data, VHva)
data = vcat(data, VLva)
data = vcat(data, Lva)
data = vcat(data, Hva)

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

save("H-L.jld2",
  "inTr", normalizeMinMax(inputTr, Normalization),
  "inVl", normalizeMinMax(inputVl, Normalization),
  "inTs", normalizeMinMax(inputTs, Normalization),
  "trTr", targtTr,
  "trVl", targtVl,
  "trTs", targtTs)
