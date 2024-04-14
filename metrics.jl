using JLD2
using Random
using Flux
using Flux.Losses

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
  vpp = VP / (VP + FP)
  vpn = VN / (FP + VN)
  f1 = 2 * ((sen * pre) / (sen + pre))

  return (pre, err, sen, esp, vpp, vpn, f1, conf)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},
  targets::AbstractArray{Bool,1}; threshold::Real=0.5)

  return confusionMatrix(broadcast(>=, outputs, threshold), targets)
end

# Pint

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
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
  targets::AbstractArray{Bool,1}; threshold::Real=0.5)

  printConfusionMatrix(broadcast(>=, outputs, threshold), targets)
end

function calculateMetrics()
  ann = load("annAndLoss.jld2", "ann")
  inTs = load("VH-VL.jld2", "inTs")
  trTs = load("VH-VL.jld2", "trTs")
  topology = load("annAndLoss.jld2", "topology")

  outputs = ann(inTs')

  println(topology)
  printConfusionMatrix(vec(outputs'), vec(trTs'))
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

function _show_metrics(acc, errorRate, sensibility, stdSensibility, specificity, precision, stdPrecision, negativePredictiveValues, f1, matrix)
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



