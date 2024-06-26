module Metrics

	export confusionMatrix, printConfusionMatrix, show_metrics
	include("boletin02.jl")
	import .ANNUtils: oneHotEncoding, classifyOutputs

	# Auxiliar functions
	function _accuracy(confusionMatrix::AbstractArray{Int64, 2})
		numerator = 0
		rows = size(confusionMatrix, 1)

		for i in 1:rows
			numerator += confusionMatrix[i, i]
		end


		denominator = sum(confusionMatrix[:, :])

		return numerator / denominator
	end

	function _errorRate(confusionMatrix::AbstractArray{Int64, 2})
		truesPredictions = 0
		sumAllConfusionMatrix = sum(confusionMatrix)
		rows = size(confusionMatrix, 1)

		for i in 1:rows
			truesPredictions += confusionMatrix[i, i]
		end

		return (sumAllConfusionMatrix - truesPredictions) / sumAllConfusionMatrix
	end

	function _sensitivity(confusionMatrix::AbstractArray{Int64, 2})
		if (confusionMatrix[2, 1] == 0 && confusionMatrix[2, 2] == 0)
			return 1
		end
		return confusionMatrix[2, 2] / (confusionMatrix[2, 1] + confusionMatrix[2, 2])
	end

	function _specificity(confusionMatrix::AbstractArray{Int64, 2})
		if (confusionMatrix[1, 1] == 0 && confusionMatrix[1, 2] == 0)
			return 1
		end
		return confusionMatrix[1, 1] / (confusionMatrix[1, 2] + confusionMatrix[1, 1])
	end

	function _precision(confusionMatrix::AbstractArray{Int64, 2})
		return confusionMatrix[2, 2] / (confusionMatrix[2, 2] + confusionMatrix[1, 2])
	end

	function _negativePredictiveValue(confusionMatrix::AbstractArray{Int64, 2})
		if (confusionMatrix[2, 1] == 0 && confusionMatrix[1, 1] == 0)
			return 1
		end
		return confusionMatrix[1, 1] / (confusionMatrix[1, 1] + confusionMatrix[2, 1])
	end

	function _f1Score(confusionMatrix::AbstractArray{Int64, 2})
		numerator = size(confusionMatrix, 1) * 2
		denominator = 0

		for element in confusionMatrix
			denominator += 1 / element
		end

		return numerator / denominator
	end

	function _print_matrix(confusionMatrix::AbstractArray{Int64, 2})
		println("Confusion matrix with ", sum(confusionMatrix), " samples")
		println("                  Prediction")
		println("                | Negative\t| Positive")
		println("Real | Negative | ", confusionMatrix[1, 1], "\t\t| ", confusionMatrix[1, 2])
		println("     | Positive | ", confusionMatrix[2, 1], "\t\t| ", confusionMatrix[2, 2])
	end

	function _print_matrix_multiclass(confusionMatrix::AbstractArray{Int64, 2})
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
				print(confusionMatrix[i, j], "\t\t| ")
			end
			println()
		end

	end


	# Boletin04_1
	# Main functions
	function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
		VP = count((outputs .+ targets) .== 2)
		VN = count((outputs .+ targets) .== 0)
		FP = count(((outputs .== false) .+ targets) .== 0)
		FN = count(((outputs .== false) .+ targets) .== 2)
		conf = [VN FP; FN VP]

		pre = (VN + VP) / (VN + VP + FN + FP)
		err = (FN + FP) / (VN + VP + FN + FP)
		sen = VP / (FN + VP)
		esp = VN / (FP + VN)
		if (VP + FP) == 0
			vpp = 0
		else
			vpp = VP / (VP + FP)
		end
		vpn = VN / (FP + VN)
		f1 = 2 * ((sen * pre) / (sen + pre))

		return (pre, err, sen, esp, vpp, vpn, f1, conf)
	end

	function confusionMatrix(outputs::AbstractArray{<:Real, 1},
		targets::AbstractArray{<:Real, 1}; threshold::Real = 0.5)
		return confusionMatrix(broadcast(>=, outputs, threshold), broadcast(>=, targets, threshold))
	end

	function confusionMatrix(outputs::AbstractArray{<:Real, 1},
		targets::AbstractArray{Bool, 1}; threshold::Real = 0.5)
		return confusionMatrix(broadcast(>=, outputs, threshold), targets)
	end

	# Boletin04_2
	# no
	#=
	function confusionMatrix(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2};
		weighted::Bool = true)

		#@assert all([in(output, unique(targets)) for output in outputs]) "The outputs must be in the targets"
		@assert size(outputs, 2) == size(targets, 2) "The size of the outputs and targets must be the same. Outputs: $(size(outputs, 2)) Targets: $(size(targets, 2))"
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
			_, _, aux_sensitivity, aux_specificity, aux_precision, aux_negativePredictiveValue,
			aux_f1Score, _ = confusionMatrix(outputs[:, class], targets[:, class])

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
	=#


	function confusionMatrix(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2}; weighted::Bool = true)

		numClasses = size(targets, 2)
		numSamples = size(targets, 1)

		if (numClasses == 1)
			return confusionMatrix(outputs[:,1], targets[:,1])
		else
			pre = zeros(numClasses)
			err = zeros(numClasses)
			sen = zeros(numClasses)
			spe = zeros(numClasses)
			vpp = zeros(numClasses)
			vpn = zeros(numClasses)
			f1 = zeros(numClasses)
			matrix = zeros(Int64, numClasses, numClasses)

			for class in 1:numClasses
			(
				pre[class], 
				err[class], 
				sen[class], 
				spe[class], 
				vpp[class], 
				vpn[class], 
				f1[class], 
				_
			) = confusionMatrix(outputs[:,class], targets[:,class])
			end

			rows, cols = size(matrix)
			for row in 1:rows
				for col in 1:cols
					matrix[row, col] = sum(outputs[:, row] .== true .& targets[:, col] .== true)
				end
			end


			fpre, ferr, fsen, fspe, fvpp, fvpn, ff1 = 0, 0, 0, 0, 0, 0, 0

			if weighted
				for class in 1:numClasses
					classInstances = sum(targets[:, class]) / numSamples

					fpre += pre[class] * classInstances
					ferr += err[class] * classInstances
					fsen += sen[class] * classInstances
					fspe += spe[class] * classInstances
					fvpp += vpp[class] * classInstances
					fvpn += vpn[class] * classInstances
					ff1 += f1[class] * classInstances
				end
			else
			fpre = sum(pre)
			ferr = sum(err)
			fsen = sum(sen)
			fspe = sum(spe)
			fvpp = sum(vpp)
			fvpn = sum(vpn)
			ff1 = sum(f1)
			end

			fpre, ferr, fsen, fspe, fvpp, fvpn, ff1 ./ numSamples

			return (fpre, ferr, fsen, fspe, fvpp, fvpn, ff1, matrix)

		end

	end

	function confusionMatrix(outputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool, 2};
		weighted::Bool = true)

		outputs = (classifyOutputs(outputs))
		return confusionMatrix(outputs, targets, weighted = weighted)
	end


	function confusionMatrix(outputs::AbstractArray{<:Any, 1}, targets::AbstractArray{<:Any, 1};
		weighted::Bool = true)

		classes = unique(targets)
		outputs = oneHotEncoding(outputs, classes)
		targets = oneHotEncoding(targets, classes)

		return confusionMatrix(outputs, targets, weighted = weighted)
	end

	function confusionMatrix(outputs::AbstractArray{<:Any, 1}, targets::AbstractArray{<:Any, 1})

		classes = unique(targets)
		outputs = oneHotEncoding(outputs, classes)
		targets = oneHotEncoding(targets, classes)

		return confusionMatrix(outputs, targets)
	end

	function printConfusionMatrix(outputs::AbstractArray{Bool,1},
		targets::AbstractArray{Bool,1})
		result = confusionMatrix(outputs, targets)
		println("Accuracy: ", result[1])
		println("Error rate: ", result[2])
		println("Recall: ", result[3])
		println("Specificity: ", result[4])
		println("Positive predictive value: ", result[5])
		println("Negative predictive value: ", result[6])
		println("F1-score: ", result[7])
		show(stdout, "text/plain", result[8])
		println("")
	end

	function printConfusionMatrix(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2};
		weighted::Bool = true)

		(accuracy, errorRate, sensitivity, specificity, precision, negativePredictiveValue, f1Score, confusion) = confusionMatrix(outputs, targets, weighted = weighted)

		println("\nAccuracy: ", accuracy)
		println("Error rate: ", errorRate)
		println("Sensitivity: ", sensitivity)
		println("Specificity: ", specificity)
		println("Precision: ", precision)
		println("Negative predictive value: ", negativePredictiveValue)
		println("F1 score: ", f1Score)
		_print_matrix_multiclass(confusion)
	end

	function printConfusionMatrix(outputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool, 2};
		weighted::Bool = true)

		outputs = outputs .> 0.5
		printConfusionMatrix(outputs, targets, weighted = weighted)
	end

	function printConfusionMatrix(outputs::AbstractArray{<:Any, 1}, targets::AbstractArray{<:Any, 1};
		weighted::Bool = true)

		classes = unique(targets)
		outputs = oneHotEncoding(outputs, classes)
		targets = oneHotEncoding(targets, classes)
		printConfusionMatrix(outputs, targets, weighted = weighted)

	end

	function show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision, negativePredictiveValues, f1, matrix)
		println("Accuracy: ", acc)
		println("Error rate: ", errorRate)
		println("Sensibility: ", sensibility)
		println("Standard deviation of sensibility: ", stdSensibility)
		println("Specificity: ", specificity)
		println("Precision: ", precision)
		println("Standard deviation of precision: ", stdPrecision)
		println("Negative predictive values: ", negativePredictiveValues)
		println("F1: ", f1)
		println("Confusion matrix: ")
		_print_matrix(matrix)
	end

	function _print_matrix(matrix)
		rows, cols = size(matrix)
		for i in 1:rows
			for j in 1:cols
				print(matrix[i, j], "\t")
			end
			println()
		end
	end

end
