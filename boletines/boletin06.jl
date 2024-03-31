module ScikitModels
using Pkg
"""Pkg.add("Statistics")"""

include("boletin02.jl");
include("boletin03.jl");
include("boletin04.jl");
using ScikitLearn
using ScikitLearn: fit!, predict
using Flux
using Flux.Losses
using Statistics
using .Metrics: confusionMatrix
using .ANNUtils: oneHotEncoding, trainClassANN
using .Overtraining: holdOut

export modelCrossValidation, set_modelHyperparameters

    @sk_import svm: SVC
    @sk_import tree: DecisionTreeClassifier
    @sk_import neighbors: KNeighborsClassifier

    function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})
        println("Model type: ", modelType)
        @assert size(inputs, 1) == size(targets, 1) "Inputs and targets must have the same number of samples"
        @assert (modelType == :SVC) || 
                (modelType == :DecissionTreeClassifier) || 
                (modelType == :KNeighborsClassifier) || 
                (modelType == :ANN) "Model must be SVC, DecissionTreeClassifier, KNeighborsClassifier or ANN"

        println("Model type: ", modelType)
        if modelType == :ANN
            topology = modelHyperparameters["topology"]
            maxEpochs = modelHyperparameters["maxEpochs"]
            maxEpochsVal = modelHyperparameters["maxEpochsVal"]
            learningRate = modelHyperparameters["learningRate"]
            transferFunctions = modelHyperparameters["transferFunctions"]
            validationRatio = modelHyperparameters["validationRatio"]
            testRatio = modelHyperparameters["testRatio"]
            numExecutions = modelHyperparameters["numExecutions"]
            
            targets = oneHotEncoding(targets)
            trainingInputs = inputs[crossValidationIndices .!= 1, :]
            trainingTargets = targets[crossValidationIndices .!= 1, :]
            testingInputs = inputs[crossValidationIndices .== 1, :]
            testingTargets = targets[crossValidationIndices .== 1, :]

            accuracyPerTraining = zeros(numExecutions)
            errorRatePerTraining = zeros(numExecutions)
            sensibilityPerTraining = zeros(numExecutions)
            specificityPerTraining = zeros(numExecutions)
            precisionPerTraining = zeros(numExecutions)
            negativePredictiveValuesPerTraining = zeros(numExecutions)
            f1PerTraining = zeros(numExecutions)
            confusionMatrixPerTraining = zeros(size(targets, 2), size(targets, 2), numExecutions)

            for numExecution in 1:numExecutions
                if validationRatio > 0.0
                    train, val, test = holdOut(size(inputs, 1), validationRatio, testRatio)

                    (bestAnn, _, _, _) = trainClassANN(topology, (inputs[train, :], targets[train, :]), 
                        validationDataset=(inputs[val, :], targets[val, :]), testDataset=(inputs[test, :], targets[test, :]), 
                        learningRate=learningRate, maxEpochs=maxEpochs, 
                        maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions)
                else
                    (bestAnn, _, _, _) = trainClassANN(topology, (trainingInputs, trainingTargets), 
                        testDataset=(testingInputs, testingTargets), 
                        learningRate=learningRate, maxEpochs=maxEpochs, 
                        maxEpochsVal=maxEpochsVal, transferFunctions=transferFunctions)
                end
                
                println(collect(bestAnn(testingInputs')'))
                println(testingTargets)

                (accuracyPerTraining[numExecution], errorRatePerTraining[numExecution], 
                    sensibilityPerTraining[numExecution], specificityPerTraining[numExecution], 
                    precisionPerTraining[numExecution], negativePredictiveValuesPerTraining[numExecution], 
                    f1PerTraining[numExecution], confusionMatrixPerTraining[:,:,numExecution]) = confusionMatrix(collect(bestAnn(testingInputs')'), testingTargets)
            end

        else
            numExecutions = modelHyperparameters["numExecutions"]

            accuracyPerTraining = zeros(numExecutions)
            errorRatePerTraining = zeros(numExecutions)
            sensibilityPerTraining = zeros(numExecutions)
            specificityPerTraining = zeros(numExecutions)
            precisionPerTraining = zeros(numExecutions)
            negativePredictiveValuesPerTraining = zeros(numExecutions)
            f1PerTraining = zeros(numExecutions)
            confusionMatrixPerTraining = zeros(size(targets, 2), size(targets, 2), numExecutions)

            for numExecution in 1:numExecutions

                if modelType == :SVC
                    possibleKernel = ["linear", "poly", "rbf", "sigmoid"]
                    C = modelHyperparameters["C"]
                    kernel = modelHyperparameters["kernel"]
                    degree = modelHyperparameters["degree"]
                    gamma = modelHyperparameters["gamma"]
                    coef0 = modelHyperparameters["coef0"]
                    @assert kernel in possibleKernel "Kernel must be linear, poly, rbf or sigmoid"

                    if kernel == "linear"
                        #@assert modelCrossValidation.hasKey("C") "In linear kernel, C must be defined"

                        model = SVC(kernel=kernel, C=C)
                    
                    elseif kernel == "poly"
                        @assert degree.hasKey("degree") && gamma.hasKey["gamma"] && coef0.hasKey["coef0"] && C.hasKey["C"] "In linear kernel, degree, gamma, coef0 and C must be defined"

                        model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0)
                    
                    elseif kernel == "rbf"
                        @assert gamma.hasKey["gamma"] && C.hasKey["C"] "In rbf kernel, gamma and C must be defined"

                        model = SVC(kernel=kernel, C=C, gamma=gamma)
                    
                    elseif kernel == "sigmoid"
                        @assert gamma.hasKey["gamma"] && coef0.hasKey["coef0"] && C.hasKey["C"] "In sigmoid kernel, gamma, coef0 and C must be defined"

                        model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
                        
                    end

                elseif modelType == :DecissionTreeClassifier
                    max_depth = modelHyperparameters["max_depth"]

                    @assert max_depth.hasKey("max_depth") "In DecisionTreeClassifier, max_depth must be defined"
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)

                elseif modelType == :KNeighborsClassifier
                    n_neighbors = modelHyperparameters["n_neighbors"]

                    @assert n_neighbors.hasKey("n_neighbors") "In KNeighborsClassifier, n_neighbors must be defined"

                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                end

                trainingInputs = inputs[crossValidationIndices .!= 1, :]
                trainingTargets = targets[crossValidationIndices .!= 1]
                testingInputs = inputs[crossValidationIndices .== 1, :]
                testingTargets = targets[crossValidationIndices .== 1]
    
                model = fit!(model, trainingInputs, trainingTargets);
    
                outputs = predict(model, testingInputs);
    
                (accuracyPerTraining[numExecution], errorRatePerTraining[numExecution], 
                    sensibilityPerTraining[numExecution], specificityPerTraining[numExecution], 
                    precisionPerTraining[numExecution], negativePredictiveValuesPerTraining[numExecution], 
                    f1PerTraining[numExecution], confusionMatrixPerTraining[:,:,numExecution]) = confusionMatrix(outputs, testingTargets)
            end

        end

        acc = mean(accuracyPerTraining)
        println(accuracyPerTraining)
        errorRate = mean(errorRatePerTraining)
        sensibility = mean(sensibilityPerTraining)
        specificity = mean(specificityPerTraining)
        precision = mean(precisionPerTraining)
        negativePredictiveValues = mean(negativePredictiveValuesPerTraining)
        f1 = mean(f1PerTraining)
        matrix = mean(confusionMatrixPerTraining, dims=3)

        return (acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)

    end


    function set_modelHyperparameters(modelType::Symbol; kernel::String="linear", C::Float64=0.0, 
                                        degree::Int64=0, gamma::Float64=0.0, 
                                        coef0::Float64=0.0, topology::Array{Int64,1}=[2,2],
                                        learningRate::Float64=0.01, validationRatio::Float64=0.2,
                                        testRatio::Float64=0.1, numExecutions::Int64=50, maxEpochs::Int64=1000,
                                        maxEpochsVal::Int64=6, transferFunctions::Array{Function,1}=[Flux.relu, Flux.sigmoid],
                                        n_neighbors::Int64=5, max_depth::Int64=6)
        @assert (modelType == :ANN) || 
                (modelType == :SVC) || 
                (modelType == :DecissionTreeClassifier) || 
                (modelType == :KNeighborsClassifier) 
                "Model must be ANN, SVC, DecissionTreeClassifier or KNeighborsClassifier"
        
        dict = Dict{String, Any}("modelType" => modelType, "kernel" => kernel)

        if modelType == :ANN
            dict["topology"] = topology
            dict["learningRate"] = learningRate
            dict["validationRatio"] = validationRatio
            dict["testRatio"] = testRatio
            dict["maxEpochs"] = maxEpochs
            dict["maxEpochsVal"] = maxEpochsVal
            dict["transferFunctions"] = transferFunctions
            @assert testRatio + validationRatio < 1.0 "The sum of testRatio and validationRatio must be less than 1"
            
        else
            println("Model type: ", modelType)
            dict["C"] = C
            dict["degree"] = degree
            dict["gamma"] = gamma
            dict["coef0"] = coef0
            dict["n_neighbors"] = n_neighbors
            dict["max_depth"] = max_depth
        end

        dict["numExecutions"] = numExecutions
        
        return dict
    end

end