using Pkg;
"""Pkg.add("Random");"""


module Overtraining

using Random;

export holdOut

    function holdOut(N::Int, P::Real)
        index = randperm(N)
        test = convert(Int, round(N*P))

        @assert (length(index[1:test]) + 
                length(index[test+1:end])) == N 
                "Length of training and test sets must be equal to N"

        return index[1:test], index[test+1:end]
    end

    function holdOut(N::Int, Pval::Real, Ptest::Real)
        index = randperm(N)
        test = convert(Int, round(N*Ptest))
        val = convert(Int, round(N*Pval))
        train = N - test - val

        @assert (length(index[1:val]) + 
                length(index[val+1:val+test]) + 
                length(index[val+test+1:end])) == N 
                "Length of training, validation and test sets must be equal to N"

        return index[1:train], index[train+1:train+val], index[train+val+1:end]
    end

end