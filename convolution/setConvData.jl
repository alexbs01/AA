using JLD2
using Images
using Random

function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
  matrix = Array{Float32,3}(undef, size(image, 1), size(image, 2), 3)
  matrix[:, :, 1] = convert(Array{Float32,2}, red.(image))
  matrix[:, :, 2] = convert(Array{Float32,2}, green.(image))
  matrix[:, :, 3] = convert(Array{Float32,2}, blue.(image))
  return matrix
end
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image))

function loadFolderImages(folderName::String, num::Int)
  isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"])
  images = []
  for fileName in readdir(folderName)
		num -= 1
		if num == 0 
			break
		end
    if isImageExtension(fileName)
      image = load(string(folderName, "/", fileName))
      # Check that they are color images
      @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
      # Add the image to the vector of images

      push!(images, image)
    end
  end
  # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
  return imageToColorArray.(images)
end

fileoutput = "ConvVHVL.jld2"
folder = "dataset/train"

num = 2000
vh = loadFolderImages("$folder/Very_High/", num)
vhT = Float32.(ones(size(vh, 1)))
vh = hcat(vh, vhT)

vl = loadFolderImages("$folder/Very_Low/", num)
vlT = Float32.(zeros(size(vl, 1)))
vl = hcat(vl, vlT)

v = vcat(vh, vl)
v = v[shuffle(1:end), :]

save("comv.jld2","im", v[:,1], "tag", v[:,2])
