using JLD2

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

ann = load_object("annAndLoss.jld2")
data = load_object("VH-VL.jld2")


