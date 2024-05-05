include("Modelos/ANN.jl")
include("Modelos/DecissionTreeClassifier.jl")
include("Modelos/SVC.jl")
include("Modelos/KNeighborsClassifier.jl")

using .ANN: execute
using .DecissionTreeClassifier: execute
using .SVC: execute
using .KNeighborsClassifier: execute
using Random

# Generara una semilla aleatoria

Random.seed!(88008)

# Cargar datos y Extraer las características de esa aproximación 

file = "datasetForAprox/VH-M-VL2_aprox3.jld2"

#ANN.execute(file)
DecissionTreeClassifier.execute(file)
#SVC.execute(file)
#KNeighborsClassifier.execute(file)