using DataFrames
using CSV
using Statistics
using GLM
function load_data(filepath::String)
    return CSV.File(filepath) |> DataFrame
end

function clean_data(df::DataFrame)
    select!(df, Not(:Date))  
    filter!(row -> all(!ismissing, row), df) 
    return df
end


function summary_statistics(df::DataFrame)
    println("Summary Statistics:")
    println("Mean Temperature: ", mean(df.Temperature))
    println("Mean Humidity: ", mean(df.Humidity))
    println("Mean Precipitation: ", mean(df.Precipitation))
    println("Mean Wind Speed: ", mean(df.WindSpeed))
    println("Mean Air Quality Index: ", mean(df.AirQualityIndex))
end

function correlation_analysis(df::DataFrame)
    println("\nCorrelation Analysis:")
    correlation_matrix = cor(Matrix(df[:, Not(:Date)]))
    println("Correlation Matrix:\n", correlation_matrix)
end

function group_by_city(df::DataFrame)
    grouped_df = groupby(df, :City)
    city_means = combine(grouped_df, 
        :Temperature => mean,
        :Humidity => mean,
        :Precipitation => mean,
        :WindSpeed => mean,
        :AirQualityIndex => mean
    )
    println("\nMean Values by City:")
    println(city_means)
end

function custom_analysis(df::DataFrame)
    function categorize_temperature(row)
        return row.Temperature > 30 ? "High" : "Low"
    end
    df[:, :Temperature_Category] = map(categorize_temperature, eachrow(df))
    println("\nData with Temperature Categories:")
    println(df)
end

function seasonal_summary(df::DataFrame)
    seasonal_summary_df = combine(groupby(df, :Season), 
        :Temperature => mean,
        :Humidity => mean,
        :Precipitation => sum,
        :WindSpeed => mean
    )
    println("\nSeasonal Summary:")
    println(seasonal_summary_df)
end

function perform_linear_regression(df::DataFrame)
    model = lm(@formula(Temperature ~ Humidity + Precipitation + WindSpeed + AirQualityIndex), df)
    println("\nLinear Regression Model Summary:")
    println(coef(model))
    println("R²: ", r2(model))
    return model
end

function make_prediction(model, humidity, precipitation, wind_speed, air_quality_index)
    example_input = DataFrame(Humidity=[humidity], Precipitation=[precipitation], WindSpeed=[wind_speed], AirQualityIndex=[air_quality_index])
    predicted_temp = predict(model, example_input)
    println("\nPredicted Temperature for the example input: ", predicted_temp)
end

df = load_data("climate_data.csv")
df = clean_data(df)

summary_statistics(df)
correlation_analysis(df)
group_by_city(df)
custom_analysis(df)
seasonal_summary(df)

model = perform_linear_regression(df)
make_prediction(model, 20, 100, 3, 200)
