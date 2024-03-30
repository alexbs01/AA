module ScikitModels
include("boletin02.jl");
include("boletin03.jl");
include("boletin04.jl");
using ScikitLearn
using ScikitLearn: fit!, predict
using Flux
using Flux.Losses
using .Metrics: confusionMatrix
using .ANNUtils: oneHotEncoding, trainClassANN
using .Overtraining: holdOut

export modelCrossValidation, set_modelHyperparameters

    @sk_import svm: SVC
    @sk_import tree: DecisionTreeClassifier
    @sk_import neighbors: KNeighborsClassifier

    modelHyperparameters = Dict("topology" => [5,3], "learningRate" => 0.01,
    "validationRatio" => 0.2, "numExecutions" => 50, "maxEpochs" => 1000,
    "maxEpochsVal" => 6);

    function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})

        @assert size(inputs, 1) == size(targets, 1) "Inputs and targets must have the same number of samples"
        @assert (model==:SVC) || (model==:DecissionTreeClassifier) || (model==:KNeighborsClassifier) || (model==:ANN) "Model must be SVC, DecissionTreeClassifier, KNeighborsClassifier or ANN"

        if modelType == :ANN
            targets = oneHotEncoding(targets)

            train, val, test = holdOut(size(inputs, 1), modelHyperparameters["validationRatio"], modelHyperparameters["testRatio"])
            topology = modelHyperparameters["topology"]
            maxEpochs = modelHyperparameters["maxEpochs"]
            maxEpochsVal = modelHyperparameters["maxEpochsVal"]
            learningRate = modelHyperparameters["learningRate"]
            transferFunctions = modelHyperparameters["transferFunctions"]

            (bestAnn, trainingLosses, validationLosses, testLosses) = trainClassANN(topology, (inputs[train, :], targets[train, :]), 
                validationDataset=(inputs[val, :], targets[val, :]), testDataset(inputs[test, :], targets[test, :]), 
                learningRate=learningRate, maxEpochs=maxEpochs, 
                maxEpochVal=maxEpochsVal, transferFunctions=transferFunctions)
            

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

                    if degree.hasKey("degree") || gamma.hasKey["gamma"] || coef0.hasKey["coef0"] 
                        println("In linear kernel, degree, gamma and coef0 will be ignored")
                    end

                    model = SVC(kernel=kernel, C=C)
                
                elseif kernel == "poly"
                    @assert degree.hasKey("degree") && gamma.hasKey["gamma"] && coef0.hasKey["coef0"] && C.hasKey["C"] "In linear kernel, degree, gamma, coef0 and C must be defined"

                    model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0)
                
                elseif kernel == "rbf"
                    @assert gamma.hasKey["gamma"] && C.hasKey["C"] "In rbf kernel, gamma and C must be defined"

                    if degree.hasKey("degree") || coef0.hasKey("coef0")
                        println("In rbf kernel, degree and coef0 will be ignored")
                    end

                    model = SVC(kernel=kernel, C=C, gamma=gamma)
                
                elseif kernel == "sigmoid"
                    @assert gamma.hasKey["gamma"] && coef0.hasKey["coef0"] && C.hasKey["C"] "In sigmoid kernel, gamma, coef0 and C must be defined"
                    
                    if degree.hasKey["degree"]
                        println("In sigmoid kernel, degree will be ignored")
                    end

                    model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
                    
                end

            elseif modelType == :DecissionTreeClassifier
                max_depth = modelHyperparameters["max_depth"]

                @assert max_depth.hasKey("max_depth") "In DecisionTreeClassifier, max_depth must be defined"
                model = DecisionTreeClassifier(max_depth=max_depth)

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

    modelHyperparameters = Dict("topology" => [5,3], "learningRate" => 0.01,
"validationRatio" => 0.2, "numExecutions" => 50, "maxEpochs" => 1000,
"maxEpochsVal" => 6);

    function set_modelHyperparameters(modelType::Symbol; kernel::String="linear", C::Float64=0.0, 
                                        degree::Int64=0, gamma::Float64=0.0, 
                                        coef0::Float64=0.0, topology::Array{Int64,1}=[2,3],
                                        learningRate::Float64=0.01, validationRatio::Float64=0.2,
                                        numExecutions::Int64=50, maxEpochs::Int64=1000,
                                        maxEpochsVal::Int64=6, transferFunctions::Array{Function,1}=[Flux.relu, Flux.sigmoid])
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
        end
        
        return dict
    end

end