module Metrics
export confusionMatrix

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
  VP = count((outputs .+ targets) .== 2)
  VN = count((outputs .+ targets) .== 0)
  FP = count(((outputs .== false) .+ targets) .== 0)
  FN = count(((outputs .== false) .+ targets) .== 2)
  conf = [VN FP; FN VP]

  pre = (VN + VP) / (VN + VP + FN + FP)
  err = (FN + FP) / (VN + VP + FN + FP)

  if FN + VP > 0
    sen = VP / (FN + VP)
  else
    sen = 0
  end

  if FP + VN > 0
    esp = VN / (FP + VN)
    vpn = VN / (FP + VN)
  else
    esp = 0
    vpn = 0
  end

  if VP + FP > 0
    vpp = VP / (VP + FP)
  else
    vpp = 0
  end

  if sen + pre > 0
    f1 = 2 * ((sen * pre) / (sen + pre))
  else
    f1 = 0
  end

  return (pre, err, sen, esp, vpp, vpn, f1, conf)
end


function confusionMatrix(outputs::AbstractArray{<:Real,1},
  targets::AbstractArray{Bool,1}; threshold::Real=0.5)

  return confusionMatrix(broadcast(>=, outputs, threshold), targets)
end

end
