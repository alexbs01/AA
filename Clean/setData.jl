include("data.jl")
import .Dataset: generateDataFile


generateDataFile("VH-VL.jld2", [("Very_High", 1.0), ("Very_Low", 0.0)])
