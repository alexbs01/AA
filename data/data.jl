module Dataset
export generateDataFile

using JLD2
using Images
using Statistics

function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
  matrix = Array{Float64,3}(undef, size(image, 1), size(image, 2), 3)
  matrix[:, :, 1] = convert(Array{Float64,2}, red.(image))
  matrix[:, :, 2] = convert(Array{Float64,2}, green.(image))
  matrix[:, :, 3] = convert(Array{Float64,2}, blue.(image))
  return matrix
end
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image))

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

function meanRGB(image::Array{Float64,3})
  [mean(image[:, :, 1]) mean(image[:, :, 2]) mean(image[:, :, 3])]
end

function stdRGB(image::Array{Float64,3})
  [std(image[:, :, 1]) std(image[:, :, 2]) std(image[:, :, 3])]
end

function imageLoader(folder::String, type::Float64, numPerClass::Integer)
  imagArr = Array{Any}(undef, 0, 7)
  for fileName in readdir(folder)
    numPerClass -= 1
    if numPerClass == 0
      break
    end
    imag = imageToColorArray(load(string(folder, fileName)))
    imagArr = vcat(imagArr, hcat(meanRGB(imag), stdRGB(imag), type))
  end
  return imagArr
end

function imageLoader2(folder::String, type::Float64, numPerClass::Integer)
  imagArr = Array{Any}(undef, 0, 9)
  for fileName in readdir(folder)
    numPerClass -= 1
    if numPerClass == 0
      break
    end
    imag = imageToColorArray(load(string(folder, fileName)))

    cord = split(fileName, "_")

    lat = cord[size(cord, 1)]
    lat = parse(Float32, String.(lat[1:end-4]))
    alt = parse(Float32, String.(cord[size(cord, 1)-1]))

    imagArr = vcat(imagArr, hcat(meanRGB(imag), stdRGB(imag), lat, alt, type))
  end
  return imagArr
end

imageToGrayArray(image::Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)))
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image))
function meanGray(image::Array{Float64,2})
  [mean(image[:, 1]) mean(image[:, 2])]
end
function stdGray(image::Array{Float64,2})
  [std(image[:, 1]) std(image[:, 2])]
end

function imageLoader3(folder::String, type::Float64, numPerClass::Integer)
  imagArr = Array{Any}(undef, 0, 13)
  for fileName in readdir(folder)
    numPerClass -= 1
    if numPerClass == 0
      break
    end
    imag = imageToColorArray(load(string(folder, fileName)))
    imagG = imageToGrayArray(load(string(folder, fileName)))

    cord = split(fileName, "_")
    lat = cord[size(cord, 1)]
    lat = parse(Float32, String.(lat[1:end-4]))
    alt = parse(Float32, String.(cord[size(cord, 1)-1]))

    imagArr = vcat(imagArr, hcat(meanRGB(imag), stdRGB(imag), meanGray(imagG), stdGray(imagG), lat, alt, type))
  end
  return imagArr
end

function generateDataFile(out::String, sources::Array{Tuple{String,Float64}}, numPerClass::Integer)
  data = Array{Any}(undef, 0, 7)
  for (source, val) in sources
    sorTR = imageLoader("dataset/train/$source/", val, numPerClass)
    #sorVa = imageLoader("dataset/val/$source/", val)
    data = vcat(data, sorTR)
    #data = vcat(data, sorVa)
    println(source)
  end
  Normalization = calculateMinMaxNormalizationParameters(Float32.(data[:, 1:6]))
  input = data[:, 1:6]
  input = Float32.(input)

  targt = data[:, 7]

  save(out,
    "in", normalizeMinMax(input, Normalization),
    "tr", targt,
    "norm", Normalization
  )
end

function generateDataFile2(out::String, sources::Array{Tuple{String,Float64}}, numPerClass::Integer)
  data = Array{Any}(undef, 0, 9)
  for (source, val) in sources
    sorTR = imageLoader2("dataset/train/$source/", val, numPerClass)
    #sorVa = imageLoader("dataset/val/$source/", val)

    data = vcat(data, sorTR)
    #data = vcat(data, sorVa)
    println(source)
  end
  Normalization = calculateMinMaxNormalizationParameters(Float32.(data[:, 1:8]))
  input = data[:, 1:8]
  input = Float32.(input)

  targt = data[:, 9]


  save(out,
    "in", normalizeMinMax(input, Normalization),
    "tr", targt,
    "norm", Normalization
  )
end

function generateDataFile3(out::String, sources::Array{Tuple{String,Float64}}, numPerClass::Integer)
  data = Array{Any}(undef, 0, 13)
  for (source, val) in sources
    sorTR = imageLoader3("dataset/train/$source/", val, numPerClass)
    #sorVa = imageLoader("dataset/val/$source/", val)

    data = vcat(data, sorTR)
    #data = vcat(data, sorVa)
    println(source)
  end
  Normalization = calculateMinMaxNormalizationParameters(Float32.(data[:, 1:12]))
  input = data[:, 1:12]
  input = Float32.(input)

  targt = data[:, 13]

  save(out,
    "in", normalizeMinMax(input, Normalization),
    "tr", targt,
    "norm", Normalization
  )
end

end


println(imageToColorArray(load("dataset/train/Very_High/27041631_5_-123.547051830273_41.5463004986268.png")));


folder = "dataset/train/Moderate/"

MeanimagArr = Array{Any}(undef, 0, 3)
StdimagArr = Array{Any}(undef, 0, 3)

for fileName in readdir(folder)
  imag = imageToColorArray(load(string(folder, fileName)))
  MeanimagArr = vcat(MeanimagArr, meanRGB(imag))
  StdimagArr = vcat(StdimagArr, stdRGB(imag))

end

println(mean(MeanimagArr, dims=1))
println(mean(StdimagArr, dims=1))