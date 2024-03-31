using Pkg;
"""Pkg.add("DelimitedFiles");
Pkg.add("Flux");
Pkg.add("Statistics");"""

using DelimitedFiles;
using Flux;
using Flux.Losses;

include("boletin02.jl");
include("boletin03.jl");
include("boletin04.jl");
include("boletin05.jl");
include("boletin06.jl");
import .ANNUtils: oneHotEncoding, calculateMinMaxNormalizationParameters, 
                calculateZeroMeanNormalizationParameters, normalizeMinMax,
                normalizeMinMax!, normalizeZeroMean, normalizeZeroMean!, 
                classifyOutputs, accuracy, buildClassANN, trainClassANN;
                
import .Overtraining: holdOut;
import .Metrics: confusionMatrix, printConfusionMatrix;
import .CrossValidation: crossvalidation, ANNCrossValidation;
import .ScikitModels: modelCrossValidation, set_modelHyperparameters;

dataset = readdlm("boletines/iris/iris.data", ',');

inputs = Float32.(dataset[:, 1:4]);
output = dataset[:, end];

function _print_matrix(matrix)
    rows, cols = size(matrix)
    for i in 1:rows
        for j in 1:cols
            print(matrix[i, j], "\t")
        end
        println()
    end
end


# BOLETIN 02
function test_oneHotEncoding()
    # ONE HOT ENCODING
    features = unique(output)
    out = oneHotEncoding(output[1:90], features[1:2])
    println("\nNormalizing multiclass with 2 features")
    println("Values: ", features[1:2])
    println("Output 1: ", out[1, :])
    println("Output 61: ", out[61, :])

    out = oneHotEncoding(output[1:90])
    println("\nNormalizing multiclass with 2 features using overload")
    println("Values: ", features[1:2])
    println("Output 1: ", out[1, :])
    println("Output 61: ", out[61, :])

    out = oneHotEncoding(output, features)
    println("\nNormalizing multiclass with more than 3 features")
    println("Values: ", features)
    println("Output 1  : ", out[1, :])
    println("Output 61 : ", out[61, :])
    println("Output 111: ", out[111, :])

    out = oneHotEncoding(output)
    println("\nNormalizing multiclass with more than 3 features using overload")
    println("Values: ", features)
    println("Output 1  : ", out[1, :])
    println("Output 61 : ", out[61, :])
    println("Output 111: ", out[111, :])
end

function test_calculateMinMaxNormalizationParameters()
    # CALCULATE NORMALIZATION PARAMETERS
    ## MIN AND MAX
    out = calculateMinMaxNormalizationParameters(inputs)
    println("\nCalculate min and max normalization parameters")
    println("Out: ", out)
    println("Out: ", out[1])
    println("Out: ", out[2][1])
end

function test_calculateZeroMeanNormalizationParameters()
    ## ZERO MEAN AND STD
    out = calculateZeroMeanNormalizationParameters(inputs)
    println("\nCalculate mean and std normalization parameters")
    println("Out: ", out)
    println("Out: ", out[1])
    println("Out: ", out[2][1])
end

function test_normalizeMinMax()
    # NORMALIZE
    ## MIN AND MAX
    inputs_copy = copy(inputs)
    println("\nNormalize min and max modifying the input")
    normalizeMinMax!(inputs_copy, calculateMinMaxNormalizationParameters(inputs_copy))
    println("Inputs normalyzed: ")
    _print_matrix(inputs_copy[1:3, :])

    inputs_copy = copy(inputs)
    normalizeMinMax!(inputs_copy)
    println("\nInputs normalyzed with overload: ")
    _print_matrix(inputs_copy[1:3, :])

    println("\nNormalize min and max withoyt modify the input")
    println("Inputs normalyzed: ")
    _print_matrix(normalizeMinMax(inputs, calculateMinMaxNormalizationParameters(inputs))[1:3, :])

    println("\nInputs normalyzed with overload: ")
    _print_matrix(normalizeMinMax(inputs)[1:3, :])

    println("\nTesting that input is not modified")
    _print_matrix(inputs[1:3, :])
end

function test_normalizeZeroMean()
    ## ZERO MEAN AND STD
    inputs_copy = copy(inputs)
    println("\nNormalize min and max modifying the input")
    normalizeZeroMean!(inputs_copy, calculateZeroMeanNormalizationParameters(inputs_copy))
    println("Inputs normalyzed: ")
    _print_matrix(inputs_copy[1:3, :])

    inputs_copy = copy(inputs)
    normalizeZeroMean!(inputs_copy)
    println("\nInputs normalyzed with overload: ")
    _print_matrix(inputs_copy[1:3, :])

    println("\nNormalize min and max withoyt modify the input")
    println("Inputs normalyzed: ")
    _print_matrix(normalizeZeroMean(inputs, calculateZeroMeanNormalizationParameters(inputs))[1:3, :])

    println("\nInputs normalyzed with overload: ")
    _print_matrix(normalizeZeroMean(inputs)[1:3, :])

    println("\nTesting that input is not modified")
    _print_matrix(inputs[1:3, :])
end

function test_classifyOutputs()
    # CLASIFY OUTPUTS
    out = inputs[1:5, 1]
    threshold = 5
    println("\nClassify outputs")
    println("Values: ", out)
    println("Out: ", classifyOutputs(out, threshold=threshold))
    println("Out: ", classifyOutputs(out))

    out = transpose(inputs[1:5, 1:4])
    println("\nClassify outputs in a matrix")
    _print_matrix(out)
    println("Out: ")
    _print_matrix(classifyOutputs(out))
end

function test_accuracy()
    # ACCURACY
    out =     Bool[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    targets = Bool[1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]
    println("\nAccuracy wiht ", size(out, 2) ," feature and ", size(out, 1), " samples")
    println("Outputs: ", out)
    println("Targets: ", targets)
    println("Accuracy: ", accuracy(out, targets))

    out =     Bool[1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1]
    targets = Bool[1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 0 1 1 1 1]
    println("\nAccuracy with ", size(out, 2) ," features and ", size(out, 1), " samples")
    println(size(out))
    println("Outputs: ", out)
    println("Targets: ", targets)
    println("Accuracy: ", accuracy(out, targets))

    out =     Bool[1 1; 1 1; 1 1; 1 1]
    targets = Bool[1 1; 1 1; 1 1; 0 1]
    println("\nAccuracy with ", size(out, 2) ," features and ", size(out, 1), " samples")
    println(size(out))
    println("Outputs: ", out)
    println("Targets: ", targets)
    println("Accuracy: ", accuracy(out, targets))

    out = inputs[1:11, 1]
    targets = Bool[1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0]
    threshold = 5
    println("\nAccuracy wiht ", size(out, 2) ," feature and ", size(out, 1), " samples")
    println("Outputs: ", out)
    println("Targets: ", targets)
    println("Accuracy: ", accuracy(out, targets, threshold=threshold))

    out = inputs[1:4, 1:4]
    targets = Bool[1 1 0 0; 1 1 0 0; 1 1 0 0; 0 0 1 1]
    threshold = 2
    println("\nAccuracy with ", size(out, 2) ," features and ", size(out, 1), " samples")
    println("Outputs: ")
    _print_matrix(out)
    println("Targets: ", targets)
    println("Accuracy: ", accuracy(out, targets, threshold=threshold))
end

function test_buildClassANN()
    println("\n",buildClassANN(1, [1, 2, 3], 3))
    println(buildClassANN(3, [2], 2))
    println(buildClassANN(4, [3, 2, 3], 2, transferFunctions=[relu, relu, relu]))
end

function test_trainClassANN()
    test_inputs = normalizeMinMax(inputs)
    test_targets = oneHotEncoding(output)
    println("\n",trainClassANN([1, 2, 3], (test_inputs, test_targets), maxEpochs=4))
    println("\n",trainClassANN([2], (test_inputs, test_targets), maxEpochs=3))
    println("\n",trainClassANN([3, 2, 3], (test_inputs, test_targets), transferFunctions=[relu, relu, relu], maxEpochs=5))

    test_targets = oneHotEncoding(output[1:90])
    println("\n",trainClassANN([2, 2, 2], (test_inputs[1:90, :], test_targets), maxEpochs=6))

    inputs_copy = normalizeMinMax(inputs)
    targets_copy = oneHotEncoding(output)
    N = size(inputs, 1)
    Pval = 0.2
    Ptest = 0.1
    train, validation, test = holdOut(N, Pval, Ptest)

    println("\nTesting with validation and test datasets")
    println("\n",trainClassANN([1, 2, 3], (inputs_copy[train, :], targets_copy[train, :]), 
        validationDataset=(inputs_copy[validation, :], targets_copy[validation, :]), 
        testDataset=(inputs_copy[test, :], targets_copy[test, :]), maxEpochs=4))

    println("\n",trainClassANN([2], (inputs_copy[train, :], targets_copy[train, :]), 
        validationDataset=(inputs_copy[validation, :], targets_copy[validation, :]), 
        testDataset=(inputs_copy[test, :], targets_copy[test, :]), maxEpochs=3))

    println("\n",trainClassANN([2], (inputs_copy[train, :], targets_copy[train, :]), 
        validationDataset=(inputs_copy[validation, :], targets_copy[validation, :]), 
        testDataset=(inputs_copy[test, :], targets_copy[test, :]), transferFunctions=[relu, relu, relu], maxEpochs=8))

    println("\nTesting with validation and test datasets with one output")
    inputs_copy = normalizeMinMax(inputs[1:90, :])
    targets_copy = oneHotEncoding(output[1:90])
    N = size(inputs[1:90], 1)
    Pval = 0.2
    Ptest = 0.1
    train, validation, test = holdOut(N, Pval, Ptest)
    println("\n",trainClassANN([2, 2, 2], (inputs_copy[train, :], test_targets[train]), 
        validationDataset=(inputs_copy[validation, :], targets_copy[validation]), 
        testDataset=(inputs_copy[test, :], targets_copy[test]), maxEpochs=6))
end

# BOLETIN 03
function test_holdOut()
    N = 10
    Ptrain = 0.8
    Pval = 0.2
    Ptest = 0.1

    train, test = holdOut(N, Ptrain)
    println("\nHold out with ", N, " samples and ", Ptrain*100,"% for training")
    println("Train: ", train)
    println("Test: ", test)

    train, validation, test = holdOut(N, Pval, Ptest)
    println("\nHold out with ", N," samples with ", Pval*100, "% for validation and ", Ptest*100 ,"% for testing")
    println("Train: ", train)
    println("Validation: ", validation)
    println("Test: ", test)
end

# BOLETIN 04

function test_confusionMatrix()
    outputs =      [true, false, true, true, false, true, false, false, true, true, true]
    outputs_real = [0.8,  0.1,   0.9,  0.7,  0.2,  0.6,  0.1,  0.2,  0.8,  0.9,  0.7]
    targets =      [true, false, true, true, false, true, true, true, false, false, false]
    threshold = 0.5

    println("Metrics for binary classification")
    printConfusionMatrix(outputs, targets)
    printConfusionMatrix(outputs_real, targets, threshold=threshold)

    outputs =      [false true false; true false false; false false true; true false false; true false false; false false true]
    outputs_real = [0.2  0.7 0.1;     0.8  0.3 0.1;     0.2  0.3 0.9;     0.7  0.3 0.1;   0.7  0.3 0.1;    0.2  0.3 0.9]
    targets =      [false true false; true false false; false false true; true false false; false true false; true false false]
    println("Metrics for multiclass classification")
    printConfusionMatrix(outputs, targets)
    printConfusionMatrix(outputs_real, targets, weighted=false)
end

# EJERCICIO 06
function test_set_modelHyperparameters()
    parameters1 = set_modelHyperparameters(:ANN)

    println(parameters1)
end

function test_modelCrossValidation()
    parameters = set_modelHyperparameters(:ANN, topology=[2, 2], maxEpochs=100, C=1.0)

    println(parameters)
    output_codified = oneHotEncoding(output)
    crossValidation = crossvalidation(output_codified, 5)

    (acc, errorRate, sensibility, specificity, precision, 
        negativePredictiveValues, f1, matrix) = modelCrossValidation(:ANN, parameters, inputs, output, crossValidation)

    println("Metrics for ANN")
    println("Accuracy: ", acc)
    println("Error rate: ", errorRate)
    println("Sensibility: ", sensibility)
    println("Specificity: ", specificity)
    println("Precision: ", precision)
    println("Negative predictive values: ", negativePredictiveValues)
    println("F1: ", f1)
    println("Confusion matrix: ")
    _print_matrix(matrix)
    
end
