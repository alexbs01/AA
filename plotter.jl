using JLD2
using Plots

# load ann 

lTr = load("annAndLoss.jld2", "lossTr")
lVl = load("annAndLoss.jld2", "lossVl")
lTs = load("annAndLoss.jld2", "lossTs")

#plot
len = length(lTr)
p = plot()
plot!(p, 1:len, lTr, label="Tr")
plot!(p, 1:len, lVl, label="Vl")
plot!(p, 1:len, lTs, label="Ts")
display(p)
