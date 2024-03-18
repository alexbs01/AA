using JLD2
using Plots

# load ann 

ann = load_object("annAndLoss.jld2")

#plot
len = length(ann[2])
p = plot()
plot!(p, 1:len, ann[2], label="Tr")
plot!(p, 1:len, ann[3], label="Vl")
plot!(p, 1:len, ann[4], label="Ts")
display(p)
