import Pkg;
Pkg.add("Statistics");
using Statistics;

function normalize_between_min_max(dataset)
    min_value = minimum(dataset, dims=1);
    max_value = maximum(dataset, dims=1);

    return (dataset .- min_value) ./ (max_value .- min_value), min_value, max_value;
end

function normalize_with_mean(dataset)
    mean = Statistics.mean(dataset, dims=1);
    std = Statistics.std(dataset, dims=1);

    return (dataset .- mean) ./ std, mean, std;
end

function normalize_with_multiclass(dataset)
    unique_values = unique(dataset);
    num_classes = length(unique_values);

    @assert num_classes > 1 "There must be at least two classes";

    if num_classes == 2
        return (==).(dataset, unique_values[1]), unique_values;
    else
        
        output = falses(length(dataset), num_classes);
        for j in 1:num_classes
            output[:, j] = (==).(dataset, unique_values[j]);
        end
        return output, unique_values;
    end
end

function denormalize_multiclass(result, unique_values)
    rows, cols = axes(result); # Get the dimensions of result
    output = fill("", rows);

    for i in rows
        for j in cols
            if result[i, j]
                output[i] = unique_values[j];
                break;
            end
        end
    end
    
    return output;
end