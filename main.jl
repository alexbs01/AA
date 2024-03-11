import Pkg
Pkg.add("JLD2")
Pkg.add("Images")
Pkg.add("Statistics")

using JLD2
using Images
using Statistics

# Functions that allow the conversion from images to Float64 arrays
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
  matrix = Array{Float64,3}(undef, size(image, 1), size(image, 2), 3)
  matrix[:, :, 1] = convert(Array{Float64,2}, red.(image))
  matrix[:, :, 2] = convert(Array{Float64,2}, green.(image))
  matrix[:, :, 3] = convert(Array{Float64,2}, blue.(image))
  return matrix
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));


# Functions to load the dataset

VLow = "dataset/train/Very_Low/"
VHigh = "dataset/train/Very_High/"



image = imageToColorArray(load("dataset/train/Very_High/27340611_5_-110.15605261856_43.68153326218.png"))

function meanRGB(image::Array{Float64,3})
  (mean(image[:, :, 1]), mean(image[:, :, 2]), mean(image[:, :, 3]))
end

function stdRGB(image::Array{Float64,3})
  (std(image[:, :, 1]), std(image[:, :, 2]), std(image[:, :, 3]))

end

function imageLoader(folder::String, type::Integer)
	imagArr = []
  for fileName in readdir(folder)
    imag = imageToColorArray(load(string(folder, fileName)))
		push!(imagArr, [meanRGB(imag) stdRGB(imag) type])
  end
	return imagArr
end

VH = imageLoader("dataset/train/Very_High/", 1)
VL = imageLoader("dataset/train/Very_Low/", 0)

vcat(VH, VL)
