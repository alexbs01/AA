include("convolution/convolucion.jl")

using Random
using .conv: convExc


Random.seed!(88008)

convExc([0], 3, 2048)
convExc([4], 3, 2048)
convExc([5], 3, 2048)
convExc([7], 3, 2048)
convExc([5; 4], 3, 2048)

convExc([0], 2, 2048)
convExc([4], 2, 2048)
convExc([5], 2, 2048)
convExc([7], 2, 2048)
convExc([7; 5], 2, 2048)