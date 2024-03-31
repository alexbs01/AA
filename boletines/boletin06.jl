module ScikitModels
include("boletin02.jl");
include("boletin03.jl");
include("boletin04.jl");
using ScikitLearn
using ScikitLearn: fit!, predict
using Flux
using Flux.Losses
using Flux.fordward
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

        @assert size(inputs, 1) == size(targets, 1) "Inputs and targets must have the same number of samples"
        @assert (model==:SVC) || (model==:DecissionTreeClassifier) || (model==:KNeighborsClassifier) || (model==:ANN) "Model must be SVC, DecissionTreeClassifier, KNeighborsClassifier or ANN"

        if modelType == :ANN
            topology = modelHyperparameters["topology"]
            maxEpochs = modelHyperparameters["maxEpochs"]
            maxEpochsVal = modelHyperparameters["maxEpochsVal"]
            learningRate = modelHyperparameters["learningRate"]
            transferFunctions = modelHyperparameters["transferFunctions"]
            validationRatio = modelHyperparameters["validationRatio"]
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
            confusionMatrixPerTraining = zeros(2, 2, numExecutions)

            for numExecution in 1:numExecutions
                if validationRatio > 0.0
                    train, val, test = holdOut(size(inputs, 1), validationRatio, testRatio)

                    (bestAnn, _, _, _) = trainClassANN(topology, (inputs[train, :], targets[train, :]), 
                        validationDataset=(inputs[val, :], targets[val, :]), testDataset(inputs[test, :], targets[test, :]), 
                        learningRate=learningRate, maxEpochs=maxEpochs, 
                        maxEpochVal=maxEpochsVal, transferFunctions=transferFunctions)
                else
                    (bestAnn, _, _, _) = trainClassANN(topology, (trainingInputs, trainingTargets), 
                        testDataset=(testingInputs, testingTargets), 
                        learningRate=learningRate, maxEpochs=maxEpochs, 
                        maxEpochVal=maxEpochsVal, transferFunctions=transferFunctions)
                end

                (accuracyPerTraining[numExecution], errorRatePerTraining[numExecution], 
                    sensibilityPerTraining[numExecution], specificityPerTraining[numExecution], 
                    precisionPerTraining[numExecution], negativePredictiveValuesPerTraining[numExecution], 
                    f1PerTraining[numExecution], confusionMatrixPerTraining[:,:,numExecution]) = confusionMatrix(collect(bestAnn(testingInputs)), testingTargets)
            end

            acc = mean(accuracyPerTraining)
            errorRate = mean(errorRatePerTraining)
            sensibility = mean(sensibilityPerTraining)
            specificity = mean(specificityPerTraining)
            precision = mean(precisionPerTraining)
            negativePredictiveValues = mean(negativePredictiveValuesPerTraining)
            f1 = mean(f1PerTraining)
            matrix = mean(confusionMatrixPerTraining, dims=3)

        else
            if modelType == :SVC
                possibleKernel = ["linear", "poly", "rbf", "sigmoid"]
                C = modelHyperparameters["C"]
                kernel = modelHyperparameters["kernel"]
                degree = modelHyperparameters["degree"]
                gamma = modelHyperparameters["gamma"]
                coef0 = modelHyperparameters["coef0"]
                @assert kernel in possibleKernel "Kernel must be linear, poly, rbf or sigmoid"

                if kernel == "linear"
                    @assert C.hasKey("C") "In linear kernel, C must be defined"

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

            (acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix) = confusionMatrix(outputs, testingTargets)
        
        end

        return (acc, errorRate, sensibility, specificity, precision, negativePredictiveValues, f1, matrix)

    end


    function set_modelHyperparameters(modelType::Symbol; kernel::String="linear", C::Float64=0.0, 
                                        degree::Int64=0, gamma::Float64=0.0, 
                                        coef0::Float64=0.0, topology::Array{Int64,1}=[2,3],
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
            if topology != [2,3]
                dict["topology"] = topology
            end
            if learningRate != 0.01
                dict["learningRate"] = learningRate
            end
            if validationRatio != 0.2
                dict["validationRatio"] = validationRatio
            end
            if testRatio != 0.1
                dict["testRatio"] = testRatio
                @assert testRatio + validationRatio < 1.0 "The sum of testRatio and validationRatio must be less than 1"
            end
            if numExecutions != 50
                dict["numExecutions"] = numExecutions
            end
            if maxEpochs != 1000
                dict["maxEpochs"] = maxEpochs
            end
            if maxEpochsVal != 6
                dict["maxEpochsVal"] = maxEpochsVal
            end
            if transferFunctions != [Flux.relu, Flux.sigmoid]
                dict["transferFunctions"] = transferFunctions
            end
        else
            if C != 0.0
                dict["C"] = C
            end
            if degree != 0
                print("Degree:")
                dict["degree"] = degree
            end
            if gamma != 0.0
                dict["gamma"] = gamma
            end
            if coef0 != 0.0
                dict["coef0"] = coef0
            end
            if n_neighbors != 5
                dict["n_neighbors"] = n_neighbors
            end
            if max_depth != 6
                dict["max_depth"] = max_depth
            end
        end
        
        return dict
    end

end