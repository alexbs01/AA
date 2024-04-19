include("data.jl")
import .Dataset: generateDataFile


generateDataFile("VH-M-VL.jld2", [("Very_High", 1.0), ("Very_Low", 0.0), ("Moderate", 0.5)])
