using CSV
using DataFrames
using Statistics

# Load the data
data = CSV.read("data.csv", DataFrame)

# Display the first few rows of the data
println("Data Preview:")
println(first(data, 5))

# Summary of the dataset
println("\nSummary of Dataset:")
println(describe(data))

# Total number of attacks
total_attacks = nrow(data)
println("\nTotal Number of Attacks: $total_attacks")

# Count number of attacks by type of bear
bear_attack_counts = combine(groupby(data, :type_of_bear), nrow => :Count)
println("\nNumber of Attacks by Type of Bear:")
println(bear_attack_counts)

# Calculate average age of victims
average_age = mean(skipmissing(data.age))
println("\nAverage Age of Victims: $average_age")

# Count number of attacks by gender
gender_counts = combine(groupby(data, :gender), nrow => :Count)
println("\nNumber of Attacks by Gender:")
println(gender_counts)

# Count attacks based on the outcome (killed/injured)
outcome_counts = combine(groupby(data, :Hunter), nrow => :Count)
println("\nNumber of Attacks by Outcome (Killed/Not Killed):")
println(outcome_counts)

# Analysis of attacks by month
month_counts = combine(groupby(data, :Month), nrow => :Count)
println("\nNumber of Attacks by Month:")
println(month_counts)

# Most dangerous locations (with more than 1 attack)
dangerous_locations = combine(groupby(data, :Location), nrow => :Count)
dangerous_locations = filter(row -> row.Count > 1, dangerous_locations)
println("\nMost Dangerous Locations (More than 1 Attack):")
println(dangerous_locations)

# Captive vs. Wild Bear Attacks
captive_counts = combine(groupby(data, :Type), nrow=> :Count)
println("\nNumber of Attacks by Bear Type (Captive vs Wild):")
println(captive_counts)

# Top descriptions of incidents (most common keywords)
description_words = split.(data.Description)
word_count = Dict{String, Int}()
for words in description_words
    for word in words
        word = lowercase(word)
        word_count[word] = get(word_count, word, 0) + 1
    end
end

data.Hunter_numeric = coalesce.(data.Hunter, 0)
age_outcome_corr = cor(data.age,data.Hunter_numeric)
println("\nCorrelation between Age and Outcome (Killed/Not Killed): $age_outcome_corr")
