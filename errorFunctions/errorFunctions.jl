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
        mse = meanSquaredError(real_values, predicted_values)
        mae = meanAbsoluteError(real_values, predicted_values)
        msle = meanSquaredLogarithmicError(real_values, predicted_values)
        rmse = rootMeanSquaredError(real_values, predicted_values)
        
        return (mse, mae, msle, rmse)
    end

    function showErrorFunctions(mse::Real, mae::Real, msle::Real, rmse::Real)
        println("Mean Squared Error: ", mse)
        println("Mean Absolute Error: ", mae)
        println("Mean Squared Logarithmic Error: ", msle)
        println("Root Mean Squared Error: ", rmse)
    end
end