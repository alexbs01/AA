module Utils
export oneHotEncoding

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

end
