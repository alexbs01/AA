module ErrorFunctions

export errorFunction

    function meanSquaredError(real_values::AbstractArray{<:Real,1}, predicted_values::AbstractArray{<:Real,1})
        @assert length(real_values) == length(predicted_values) "real_values and predicted_values must have the same length"
        return sum((real_values .- predicted_values).^2) / length(real_values)
    end

    function meanAbsoluteError(real_values::AbstractArray{<:Real,1}, predicted_values::AbstractArray{<:Real,1})
        @assert length(real_values) == length(predicted_values) "real_values and predicted_values must have the same length"
        return sum(abs.(real_values .- predicted_values)) / length(real_values)
    end

    function meanSquaredLogarithmicError(real_values::AbstractArray{<:Real,1}, predicted_values::AbstractArray{<:Real,1})
        @assert length(real_values) == length(predicted_values) "real_values and predicted_values must have the same length"
        return sum((log.(1 .+ real_values) .- log.(1 .+ predicted_values)).^2) / length(real_values)
    end

    function rootMeanSquaredError(real_values::AbstractArray{<:Real,1}, predicted_values::AbstractArray{<:Real,1})
        @assert length(real_values) == length(predicted_values) "real_values and predicted_values must have the same length"
        return sqrt(sum((real_values .- predicted_values).^2) / length(real_values))
    end

    function errorFunction(real_values::AbstractArray{<:Real,1}, predicted_values::AbstractArray{<:Real,1})
        println("Mean Squared Error: ", meanSquaredError(real_values, predicted_values))
        println("Mean Absolute Error: ", meanAbsoluteError(real_values, predicted_values))
        println("Mean Squared Logarithmic Error: ", meanSquaredLogarithmicError(real_values, predicted_values))
        println("Root Mean Squared Error: ", rootMeanSquaredError(real_values, predicted_values))
    end
end