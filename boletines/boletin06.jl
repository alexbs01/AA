module ScikitModels

export modelCrossValidation, set_modelHyperparameters

    using ScikitLearn
    using ScikitLearn: fit!, predict
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

        return model

    end

    function set_modelHyperparameters(kernel::String; C::Float64=0.0, 
                                        degree::Int64=0, gamma::Float64=0.0, 
                                        coef0::Float64=0.0)
        @assert kernel in ["linear", "poly", "rbf", "sigmoid"] "Kernel must be linear, poly, rbf or sigmoid"
        dict = Dict{String, Any}("kernel" => kernel)

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
        return dict
    end

end