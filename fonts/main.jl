using Pkg
"""Pkg.add("DelimitedFiles")
Pkg.add("Flux")
Pkg.add("Statistics")
Pkg.add("Random")
Pkg.add("ScikitLearn")"""


using DelimitedFiles
using Flux
using Flux.Losses
using Statistics
using Random
using ScikitLearn

include("test.jl");

println("Input size: ",size(inputs));
test02 = ["oneHotEncoding", "calculateMinMaxNormalizationParameters", 
        "calculateZeroMeanNormalizationParameters", "normalizeMinMax",
        "normalizeZeroMean", "classifyOutputs", "accuracy", 
        "buildClassANN", "trainClassANN"]

test03 = ["holdOut"]

test04 = ["confusionMatrix"]

test05 = ["crossvalidation", "ANNCrossvalidation"]

test06 = ["set_modelHyperparameters", "modelCrossValidation"]
        
tests = ["trainClassANN"]

for test in test05
    test_function = Symbol("test_", test) # Convert string to symbol (a function name)
    if @isdefined test_function # Check if the function exists
        eval(test_function)() # Call the function
    else
        println("Test $test not found")
    end
end
