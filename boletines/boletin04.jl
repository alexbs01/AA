module Metrics

export confusionMatrix, printConfusionMatrix
    include("boletin02.jl")
    import .ANNUtils: oneHotEncoding

    # Auxiliar functions
    function _accuracy(confusionMatrix::AbstractArray{Int64,2})
        numerator = 0
        rows = size(confusionMatrix, 1)

        for i in 1:rows
            numerator += confusionMatrix[i,i]
        end

        
        denominator = sum(confusionMatrix[:, :])

        if denominator <= 0
            println("DDDDDDDDDDDD")
            println(denominator)
        end

        return numerator / denominator
    end

    function _errorRate(confusionMatrix::AbstractArray{Int64,2})
        truesPredictions = 0
        sumAllConfusionMatrix = sum(confusionMatrix)
        rows = size(confusionMatrix, 1)
        
        for i in 1:rows
            truesPredictions += confusionMatrix[i,i]
        end

        return (sumAllConfusionMatrix - truesPredictions) / sumAllConfusionMatrix
    end

    function _sensitivity(confusionMatrix::AbstractArray{Int64,2})
        return confusionMatrix[2,2] / (confusionMatrix[2,1] + confusionMatrix[2,2])
    end

    function _specificity(confusionMatrix::AbstractArray{Int64,2})
        return confusionMatrix[1,1] / (confusionMatrix[1,2] + confusionMatrix[1,1])
    end

    function _precision(confusionMatrix::AbstractArray{Int64,2})
        return confusionMatrix[2,2] / (confusionMatrix[2,2] + confusionMatrix[1,2])
    end

    function _negativePredictiveValue(confusionMatrix::AbstractArray{Int64,2})
        return confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[2,1])
        
    end

    function _f1Score(confusionMatrix::AbstractArray{Int64,2})
        numerator = size(confusionMatrix, 1) * 2
        denominator = 0

        for element in confusionMatrix
            denominator += 1 / element
        end

        return numerator / denominator
    end

    function _print_matrix(confusionMatrix::AbstractArray{Int64,2})
        println("Confusion matrix with ", sum(confusionMatrix), " samples")
        println("                  Prediction")
        println("                | Negative\t| Positive")
        println("Real | Negative | ", confusionMatrix[1,1], "\t\t| ", confusionMatrix[1,2])
        println("     | Positive | ", confusionMatrix[2,1], "\t\t| ", confusionMatrix[2,2])
    end

    function _print_matrix_multiclass(confusionMatrix::AbstractArray{Int64,2})
        numClasses = size(confusionMatrix, 1)
        letter = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
        println("Confusion matrix with ", sum(confusionMatrix), " samples")
        println("                  Prediction")
        print("              ")
        for i in 1:numClasses
            print("|       $(letter[i])   \t")
        end
        println("|")

        for i in 1:numClasses
            print("Real |      $(letter[i]) | ")
            for j in 1:numClasses
                print(confusionMatrix[i,j], "\t\t| ")
            end
            println()
        end
        
    end

# Boletin04_1
# Main functions
    function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
        """
                        Prediction
                        | Negative | Positive
        Real| Negative  |   TN          FP
            | Positive  |   FN          TP
        """
        @assert length(outputs) == length(targets) "The length of the outputs and targets must be the same"
        matrix = zeros(Int64, 2, 2)

        num_trues = outputs .== targets
        num_falses = outputs .!= targets

        matrix[1,1] = sum(outputs[num_trues] .== false)  # True negatives
        matrix[1,2] = sum(outputs[num_falses] .== true)  # False positives
        matrix[2,1] = sum(outputs[num_falses] .== false) # False negatives
        matrix[2,2] = sum(outputs[num_trues] .== true)   # True positives

        println("B")
        println(matrix)

        accuracy = _accuracy(matrix)
        errorRate = _errorRate(matrix)
        sensitivity = _sensitivity(matrix)
        specificity = _specificity(matrix)
        precision = _precision(matrix)
        negativePredictiveValue = _negativePredictiveValue(matrix)
        f1Score = _f1Score(matrix)

        println("\n\n")

        return (accuracy, errorRate, sensitivity, specificity, precision, negativePredictiveValue, f1Score, matrix)
    end


    function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};
                            threshold::Real=0.5)

        outputs = (outputs .> threshold)

        return confusionMatrix(outputs, targets)
    end

    function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
        (accuracy, errorRate, sensitivity, specificity, precision, negativePredictiveValue, f1Score, confusion) = confusionMatrix(outputs, targets)
        
        println("\nAccuracy: ", accuracy)
        println("Error rate: ", errorRate)
        println("Sensitivity: ", sensitivity)
        println("Specificity: ", specificity)
        println("Precision: ", precision)
        println("Negative predictive value: ", negativePredictiveValue)
        println("F1 score: ", f1Score)
        _print_matrix(confusion)
    end

    function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; 
                                threshold::Real=0.5)

        (accuracy, errorRate, sensitivity, specificity, precision, negativePredictiveValue, f1Score, confusion) = confusionMatrix(outputs, targets, threshold=threshold)

        println("\nAccuracy: ", accuracy)
        println("Error rate: ", errorRate)
        println("Sensitivity: ", sensitivity)
        println("Specificity: ", specificity)
        println("Precision: ", precision)
        println("Negative predictive value: ", negativePredictiveValue)
        println("F1 score: ", f1Score)
        _print_matrix(confusion)
    end

    # Boletin04_2
    function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2};
        weighted::Bool=true)

        #@assert all([in(output, unique(targets)) for output in outputs]) "The outputs must be in the targets"
        @assert size(outputs, 2) == size(targets, 2) "The size of the outputs and targets must be the same"
        @assert size(outputs, 2) != 2 "The output and target must have more than two features. Outputs: $(size(outputs, 2)) Targets: $(size(targets, 2))"

        numSamples = size(targets, 1)
        numClasses = size(targets, 2)
        matrix = zeros(Int64, numClasses, numClasses)

        rows, cols = size(matrix)
        for row in 1:rows
            for col in 1:cols
                matrix[row, col] = sum(outputs[:, row] .== true .& targets[:, col] .== true)
            end
        end

        accuracy = _accuracy(matrix)
        errorRate = _errorRate(matrix)

        sensitivity, specificity, precision, negativePredictiveValue, f1Score = fill(zeros(numClasses), 5)

        for class in 1:numClasses
            _, _, aux_sensitivity, aux_specificity, aux_precision, aux_negativePredictiveValue, aux_f1Score, _ = confusionMatrix(outputs[:, class], targets[:, class])
        
            sensitivity[class] = aux_sensitivity
            specificity[class] = aux_specificity
            precision[class] = aux_precision
            negativePredictiveValue[class] = aux_negativePredictiveValue
            f1Score[class] = aux_f1Score
        end

        finalSensitivity, finalSpecificity, finalPrecision, finalNegativePredictiveValue, finalF1Score = fill(0.0, 5)

        if weighted
            for class in 1:numClasses
                classInstances = sum(targets[:, class]) / numSamples
                finalSensitivity += sensitivity[class] * classInstances
                finalSpecificity += specificity[class] * classInstances
                finalPrecision += precision[class] * classInstances
                finalNegativePredictiveValue += negativePredictiveValue[class] * classInstances
                finalF1Score += f1Score[class] * classInstances
            end
        else
            finalSensitivity = sum(sensitivity)
            finalSpecificity = sum(specificity)
            finalPrecision = sum(precision)
            finalNegativePredictiveValue = sum(negativePredictiveValue)
            finalF1Score = sum(f1Score)
        end
        
        finalSensitivity, finalSpecificity, finalPrecision, finalNegativePredictiveValue, finalF1Score ./ numSamples

        return (accuracy, errorRate, finalSensitivity, finalSpecificity, finalPrecision, finalNegativePredictiveValue, finalF1Score, matrix)
    end

    function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
        weighted::Bool=true)

        return confusionMatrix(outputs .> 0.5, targets, weighted=weighted)
    end


    function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1};
        weighted::Bool=true)

        classes = unique(targets)
        outputs = oneHotEncoding(outputs, classes)
        targets = oneHotEncoding(targets, classes)
        
        return confusionMatrix(outputs, targets, weighted=weighted) 
    end

    function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2};
        weighted::Bool=true)

        (accuracy, errorRate, sensitivity, specificity, precision, negativePredictiveValue, f1Score, confusion) = confusionMatrix(outputs, targets, weighted=weighted)

        println("\nAccuracy: ", accuracy)
        println("Error rate: ", errorRate)
        println("Sensitivity: ", sensitivity)
        println("Specificity: ", specificity)
        println("Precision: ", precision)
        println("Negative predictive value: ", negativePredictiveValue)
        println("F1 score: ", f1Score)
        _print_matrix_multiclass(confusion)
    end

    function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
        weighted::Bool=true)

        outputs = outputs .> 0.5
        printConfusionMatrix(outputs, targets, weighted=weighted)
    end

    function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1};
        weighted::Bool=true)
        
        classes = unique(targets)
        outputs = oneHotEncoding(outputs, classes)
        targets = oneHotEncoding(targets, classes)
        printConfusionMatrix(outputs, targets, weighted=weighted)

    end

end