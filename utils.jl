


"""
    Utils

A collection of utility functions for rapid development in Julia.

Developed by: Dunamix (Tech with Dunamix)

This module includes various functions for data manipulation, statistical analysis, and other useful operations,
aimed at making Julia programming more efficient and effective. The functions cater to a wide range of use cases,
from basic data processing to more complex mathematical computations.
"""

module Utils

using Random
using Statistics
using Dates
using CSV
using DataFrames
using LinearAlgebra
using Serialization

"""
    random_float(min::Float64, max::Float64)

Generates a random floating-point number between the specified `min` and `max` values.
"""
function random_float(min::Float64, max::Float64)
    rand() * (max - min) + min
end

"""
    dict_to_dataframe(nested_dict::Dict)

Converts a nested dictionary to a DataFrame.
"""
function dict_to_dataframe(nested_dict::Dict)
    DataFrame(nested_dict)
end

"""
    mode(data::AbstractVector)

Returns the mode (most frequent value) of the given data.
"""
function mode(data::AbstractVector)
    counts = countmap(data)
    argmax(values(counts))
end

"""
    remove_duplicates(arr::AbstractVector)

Removes duplicate elements from the array and returns a unique array.
"""
function remove_duplicates(arr::AbstractVector)
    unique(arr)
end

"""
    normalize(arr::AbstractVector)

Normalizes the elements of the array to a range between 0 and 1.
"""
function normalize(arr::AbstractVector)
    min_val = minimum(arr)
    max_val = maximum(arr)
    (arr .- min_val) ./ (max_val - min_val)
end

"""
    create_time_series(df::DataFrame, time_col::Symbol, value_col::Symbol)

Creates a time series object from the specified time and value columns in the DataFrame.
"""
function create_time_series(df::DataFrame, time_col::Symbol, value_col::Symbol)
    TimeArray(df[!, time_col], df[!, value_col])
end

"""
    skewness(data::AbstractVector)

Calculates the skewness of the data, indicating the asymmetry of the distribution.
"""
function skewness(data::AbstractVector)
    m = mean(data)
    n = length(data)
    sum((data .- m).^3) / (n * std(data)^3)
end

"""
    kurtosis(data::AbstractVector)

Calculates the kurtosis of the data, describing the tailedness of the distribution.
"""
function kurtosis(data::AbstractVector)
    m = mean(data)
    n = length(data)
    sum((data .- m).^4) / (n * std(data)^4) - 3
end

"""
    save_to_binary_file(variable, file_path::String)

Saves the specified variable to a binary file at the given file path.
"""
function save_to_binary_file(variable, file_path::String)
    open(file_path, "w") do file
        serialize(file, variable)
    end
end

"""
    load_from_binary_file(file_path::String)

Loads a variable from a binary file located at the given file path.
"""
function load_from_binary_file(file_path::String)
    deserialize(open(file_path, "r"))
end

"""
    dataframe_to_nested_dict(df::DataFrame)

Converts a DataFrame to a nested dictionary format.
"""
function dataframe_to_nested_dict(df::DataFrame)
    Dict(row[!, 1] => row[!, 2:end] for row in eachrow(df))
end

"""
    random_sample(arr::AbstractVector, n::Int)

Returns a random sample of size `n` from the provided array.
"""
function random_sample(arr::AbstractVector, n::Int)
    rand(arr, n)
end

"""
    flatten(nested::AbstractVector)

Flattens a nested array into a single-dimensional array.
"""
function flatten(nested::AbstractVector)
    vcat(nested...)
end

"""
    approx_equal(arr1::AbstractVector, arr2::AbstractVector; tol::Float64 = 1e-5)

Checks if two arrays are approximately equal within a given tolerance.
"""
function approx_equal(arr1::AbstractVector, arr2::AbstractVector; tol::Float64 = 1e-5)
    all(abs.(arr1 .- arr2) .< tol)
end

"""
    transpose_dataframe(df::DataFrame)

Transposes the given DataFrame.
"""
function transpose_dataframe(df::DataFrame)
    DataFrame(permutedims(Matrix(df)))
end

"""
    to_snake_case(s::String)

Converts a string to snake_case format.
"""
function to_snake_case(s::String)
    lowercase(replace(s, r" " => "_" ))
end

"""
    weighted_average(values::AbstractVector, weights::AbstractVector)

Calculates the weighted average of the provided values using the specified weights.
"""
function weighted_average(values::AbstractVector, weights::AbstractVector)
    sum(values .* weights) / sum(weights)
end

"""
    fibonacci(n::Int)

Returns the n-th Fibonacci number.
"""
function fibonacci(n::Int)
    a, b = 0, 1
    for _ in 2:n
        a, b = b, a + b
    end
    b
end

"""
    get_nested_keys(nested_dict::Dict)

Retrieves all keys from a nested dictionary.
"""
function get_nested_keys(nested_dict::Dict)
    keys_list = []
    for (k, v) in nested_dict
        push!(keys_list, k)
        if isa(v, Dict)
            append!(keys_list, get_nested_keys(v))
        end
    end
    keys_list
end

"""
    random_date(start_date::Date, end_date::Date)

Generates a random date between the specified start and end dates.
"""
function random_date(start_date::Date, end_date::Date)
    days_range = Date(end_date) - Date(start_date)
    random_days = rand(0:days_range.value)
    start_date + Day(random_days)
end

"""
    weighted_median(data::AbstractVector, weights::AbstractVector)

Calculates the weighted median of the provided data using the specified weights.
"""
function weighted_median(data::AbstractVector, weights::AbstractVector)
    sorted_indices = sortperm(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = cumsum(sorted_weights)
    total_weight = sum(weights)
    midpoint = total_weight / 2
    return sorted_data[findfirst(cumulative_weights .>= midpoint)]
end

"""
    cumulative_distribution(data::AbstractVector)

Generates the cumulative distribution of the provided data.
"""
function cumulative_distribution(data::AbstractVector)
    sorted_data = sort(data)
    n = length(sorted_data)
    return [(sorted_data[i], i / n) for i in 1:n]
end

"""
    string_to_vector(s::String)

Converts a string into a vector of its characters.
"""
function string_to_vector(s::String)
    return [c for c in s]
end

"""
    partition(array::AbstractVector, n::Int)

Partitions the array into chunks of size `n`.
"""
function partition(array::AbstractVector, n::Int)
    return [array[i:min(i + n - 1, end)] for i in 1:n:length(array)]
end

"""
    cartesian_product(sets::Vector{Vector})

Calculates the Cartesian product of a collection of sets (arrays).
"""
function cartesian_product(sets::Vector{Vector})
    return collect(Iterators.product(sets...))
end

"""
    matrix_inverse(A::AbstractMatrix)

Computes the inverse of the given matrix.
"""
function matrix_inverse(A::AbstractMatrix)
    return inv(A)
end

"""
    find_duplicates(array::AbstractVector)

Identifies duplicate elements in the array and returns their counts.
"""
function find_duplicates(array::AbstractVector)
    counts = countmap(array)
    return filter(kv -> kv[2] > 1, counts)
end

end  # module Utils
