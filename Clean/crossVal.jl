module CrossValidation
export crossvalidation

using Random
include("utils.jl")
import .Utils: oneHotEncoding

function crossvalidation(N::Int64, k::Int)
  auxVector = 1:k

  repeats = Int64(ceil(N / k))
  fullVector = repeat(auxVector, outer=repeats)
  fullVector = fullVector[1:N]

  shuffle!(fullVector)

  return fullVector
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int)
  indexes = zeros(Int64, size(targets, 1))

  posInstances = findall(targets)
  posGroups = crossvalidation(size(posInstances, 1), k)
  indexes[posInstances] .= posGroups

  negInstances = findall(.~targets)
  negGroups = crossvalidation(size(negInstances, 1), k)
  indexes[negInstances] .= negGroups

  return indexes
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int)
  indexes = zeros(Int64, size(targets, 1))
  for numClass = 1:size(targets, 2)
    class = targets[:, numClass]
    numElems = sum(class)

    @assert (numElems >= k) "El número de elementos de cada clase debe ser mayor que k"

    groups = crossvalidation(numElems, k)

    instances = findall(class)
    indexes[instances] .= groups
  end

  return indexes
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int)
  classes = unique(targets)

  @assert (length(classes) > 1) "Es necesario un mínimo de dos clases"

  if length(classes) == 2
    return crossvalidation((targets .== classes[1]), k)
  else

    return crossvalidation(oneHotEncoding(targets), k)
  end
end

end
