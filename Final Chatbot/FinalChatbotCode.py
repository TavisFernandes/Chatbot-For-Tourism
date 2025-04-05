import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('RawData.csv')

# Function to map user abstract preferences to concrete metrics
def map_preferences(preference):
    if preference.lower() == "best for business":
        return {
            "min_innovation_index": 70,
            "min_startup_ecosystem_score": 3,
            "min_gdp": 1.0  # GDP in trillions USD
        }
    elif preference.lower() == "best standard of living":
        return {
            "min_life_expectancy": 75,
            "min_healthcare_quality": 70,
            "max_cost_of_living": 150,
            "min_employment_rate": 70
        }
    elif preference.lower() == "best lifestyle":
        return {
            "min_happiness_index": 7,
            "min_work_life_balance_score": 70,
            "min_life_expectancy": 75,
            "max_crime_rate": 100  # Lower crime is better
        }
    else:
        return {}

# Function to filter countries based on mapped preferences
def suggest_countries(data, preferences):
    filtered_data = data

    if "min_innovation_index" in preferences:
        filtered_data = filtered_data[filtered_data['Innovation Index'] >= preferences['min_innovation_index']]

    if "min_startup_ecosystem_score" in preferences:
        filtered_data = filtered_data[filtered_data['Startup Ecosystem Score'] >= preferences['min_startup_ecosystem_score']]

    if "min_gdp" in preferences:
        filtered_data = filtered_data[filtered_data['GDP (in USD Trillions)'] >= preferences['min_gdp']]

    if "min_life_expectancy" in preferences:
        filtered_data = filtered_data[filtered_data['Life Expectancy (years)'] >= preferences['min_life_expectancy']]

    if "min_healthcare_quality" in preferences:
        filtered_data = filtered_data[filtered_data['Healthcare Quality Index'] >= preferences['min_healthcare_quality']]

    if "max_cost_of_living" in preferences:
        filtered_data = filtered_data[filtered_data['Cost of Living (Index)'] <= preferences['max_cost_of_living']]

    if "min_employment_rate" in preferences:
        filtered_data = filtered_data[filtered_data['Employment Rate (%)'] >= preferences['min_employment_rate']]

    if "min_happiness_index" in preferences:
        filtered_data = filtered_data[filtered_data['Happiness Index'] >= preferences['min_happiness_index']]

    if "min_work_life_balance_score" in preferences:
        filtered_data = filtered_data[filtered_data['Work-Life Balance Score'] >= preferences['min_work_life_balance_score']]

    if "max_crime_rate" in preferences:
        filtered_data = filtered_data[filtered_data['Crime Rate (per 100,000)'] <= preferences['max_crime_rate']]

    return filtered_data[['Country', 'Innovation Index', 'Life Expectancy (years)', 'Cost of Living (Index)', 'Employment Rate (%)', 'Happiness Index']]

# Function to take user input for abstract preferences
def get_user_preference():
    preference = input("Enter your preference (e.g., 'Best for Business', 'Best Standard of Living', 'Best Lifestyle'): ")
    return preference

# Get user preference through input
user_preference = get_user_preference()

# Map abstract preferences to concrete metrics
preferences = map_preferences(user_preference)

# Filter the data based on mapped preferences
suggested_countries = suggest_countries(data, preferences)

# Display the suggested countries
print("\nSuggested Countries Based on Your Preference:")
print(suggested_countries)

# Graphical Analysis: Plot a comparison of countries
plt.figure(figsize=(10, 6))
sns.barplot(x='Country', y='Happiness Index', data=suggested_countries)
plt.title('Happiness Index of Suggested Countries')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
