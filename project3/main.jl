# analyzing laptop prices to make predicitions

using DataFrames
using CSV
using Statistics
using GLM

# Load the data from the CSV file
function load_data(file_path::String)
    df = CSV.File(file_path) |> DataFrame
    return df
end

# Data Cleaning and type conversion
function clean_data(df::DataFrame)
    # Ensure 'PrimaryStorage' column exists before trying to manipulate it
    if haskey(df, :PrimaryStorage)
        # Convert empty strings to 0 and then to Int
        df.PrimaryStorage .= replace(df.PrimaryStorage, "" => "0")  # Replace empty strings with "0"
        
        # Convert to Int only if it's not already an integer
        if eltype(df.PrimaryStorage) != Int64
            df.PrimaryStorage .= parse.(Int, df.PrimaryStorage)
        end
    else
        println("Warning: 'PrimaryStorage' column not found in the DataFrame.")
    end

    # Ensure all relevant columns are of the correct type
    df.Inches = Float64.(df.Inches)  # Convert to Float64
    df.Ram = Int.(df.Ram)  # Ensure it's Int
    df.Weight = Float64.(df.Weight)  # Convert to Float64
    df.Price_euros = Float64.(df.Price_euros)  # Convert to Float64
    df.CPU_freq = Float64.(df.CPU_freq)  # Convert to Float64
end

# Summary statistics
function summarize_data(df::DataFrame)
    println("\nSummary statistics of numeric columns:")
    println(describe(df))
end

# Price Range Analysis
function price_range_analysis(df::DataFrame)
    min_price = minimum(df.Price_euros)
    max_price = maximum(df.Price_euros)
    average_price = mean(df.Price_euros)
    
    println("\nPrice Range:")
    println("Minimum Price: €$min_price")
    println("Maximum Price: €$max_price")
    println("Average Price: €$average_price")
end

# Correlation Analysis
function correlation_analysis(df::DataFrame)
    # Ensure columns exist before computing correlation
    if all(haskey(df, col) for col in [:Ram, :CPU_freq, :Weight, :Price_euros])
        correlation_matrix = cor(Matrix(df[:, [:Ram, :CPU_freq, :Weight, :Price_euros]]))
        println("\nCorrelation Matrix:")
        println(correlation_matrix)
    else
        println("Warning: One or more correlation columns are missing.")
    end
end

# Grouping Analysis
function grouping_analysis(df::DataFrame)
    group_data = groupby(df, :Company)
    # Ensure the grouping is valid and mean can be calculated
    grouped_means = combine(group_data, 
                            nrow => :Count, 
                            mean => [:Price_euros, :Ram, :CPU_freq])
    
    println("\nGrouped Analysis by Company:")
    println(grouped_means)
end

# Linear Regression Model
function linear_regression(df::DataFrame)
    # Ensure the relevant columns exist for regression
    if all(haskey(df, col) for col in [:Price_euros, :Ram, :CPU_freq])
        model = lm(@formula(Price_euros ~ Ram + CPU_freq), df)
        println("\nRegression Model Summary:")
        println(summary(model))
        println("\nModel Coefficients:")
        println(coef(model))
        return model
    else
        println("Warning: One or more regression columns are missing.")
        return nothing
    end
end

# Prediction Function
function predict_price(model, ram::Int, cpu_freq::Float64)
    if model !== nothing
        predicted_price = predict(model, DataFrame(Ram=[ram], CPU_freq=[cpu_freq]))
        println("\nPredicted Price for a laptop with $ram GB RAM and $cpu_freq GHz CPU frequency:")
        println("€$(predicted_price[1])")
    else
        println("Model is not available for prediction.")
    end
end

# Main analysis function to execute all steps
function analyze_laptops(file_path::String)
    df = load_data(file_path)
    clean_data(df)
    summarize_data(df)
    price_range_analysis(df)
    correlation_analysis(df)
    grouping_analysis(df)
    
    # Uncomment this when you're ready to use the linear regression model
    model = linear_regression(df)
    
    # Example Prediction
    best_ram = 16  # User's desired RAM
    best_cpu_freq = 3.0  # User's desired CPU frequency
    predict_price(model, best_ram, best_cpu_freq)
end

# Run the analysis
analyze_laptops("laptops.csv")
