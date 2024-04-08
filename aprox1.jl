""" Primera Aproximacion """

using Random
include("metrics.jl")

# Generara una semilla aleatoria

Random.seed!(88008)

# Cargar datos y Extraer las características de esa aproximación 

inTr = load("VH-VL.jld2", "inTr")
inVl = load("VH-VL.jld2", "inVl")
inTs = load("VH-VL.jld2", "inTs")
trTr = load("VH-VL.jld2", "trTr")
trVl = load("VH-VL.jld2", "trVl")
trTs = load("VH-VL.jld2", "trTs")

# Llamar a la función modelCrossValidation para realizar validación cruzada con
# distintos modelos y configuraciones de parámetros. 





# ANN 
topology = [3, 2]

ann = trainClassANN(
    topology,
    (inTr, trTr'),
    (inVl, trVl'),
    (inTs, trTs'))




save("annAndLoss.jld2", "topology", topology, "ann", ann[1], "lossTr", ann[2], "lossVl", ann[3], "lossTs", ann[4])

#SVC
include("Modelos/SVC.jl")

#DecisionTreeClassifier
include("Modelos/DecissionTreeClassifier.jl")

#KNeighborsClassifier
include("Modelos/KNeighborsClassifier.jl")



