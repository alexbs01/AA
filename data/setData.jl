include("data.jl")
import .Dataset: generateDataFile, generateDataFile2, generateDataFile3


#generateDataFile("VH-VL.jld2", [("Very_High", 1.0), ("Very_Low", 0.0)], 3000)
generateDataFile3("VH-M-VL3.jld2", [("Very_High", 1.0), ("Very_Low", 0.0), ("Moderate", 0.5)], 3000)