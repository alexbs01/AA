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

file = "datasetForAprox/VH-VL_aprox1.jld2"

#ANN.execute(file, false)

print("\n\n\n\n DecissionTreeClassifie \n\n\n\n")


#DecissionTreeClassifier.execute(file, false)

print("\n\n\n\n SVC \n\n\n\n")
SVC.execute(file, false)

print("\n\n\n\n KNeighborsClassifier \n\n\n\n")
#KNeighborsClassifier.execute(file, false)


