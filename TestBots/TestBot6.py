import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
file_path = "RawData.csv"
data = pd.read_csv(file_path)

# Display available columns for user reference
print("Available columns:")
print(data.columns.tolist())

# Suggest Places Function
def suggest_places(data):
    print("\nEnter your preferences (or press Enter to skip):")
    region = input("Region (e.g., Asia, Europe, South America): ").strip()
    max_cost = input("Max Cost of Living Index (numeric): ").strip()
    min_employment = input("Min Employment Rate (%) (numeric): ").strip()
    max_crime = input("Max Crime Severity Index (numeric): ").strip()
    min_environment = input("Min Environmental Performance Index (numeric): ").strip()

    filtered_data = data.copy()
    if region:
        filtered_data = filtered_data[filtered_data["Region"].str.contains(region, case=False)]
    if max_cost:
        filtered_data = filtered_data[filtered_data["Cost of Living (Index)"] <= float(max_cost)]
    if min_employment:
        filtered_data = filtered_data[filtered_data["Employment Rate (%)"] >= float(min_employment)]
    if max_crime:
        filtered_data = filtered_data[filtered_data["Crime Severity Index"] <= float(max_crime)]
    if min_environment:
        filtered_data = filtered_data[filtered_data["Environmental Performance Index"] >= float(min_environment)]

    if not filtered_data.empty:
        print("\nSuggested Places:")
        print(filtered_data[["Region", "Country", "State/Province", "Cost of Living (Index)",
                             "Employment Rate (%)", "Crime Severity Index",
                             "Environmental Performance Index"]])
    else:
        print("\nNo places match your preferences.")

# Compare Multiple Places Function
def compare_multiple_places(data):
    n = int(input("How many places do you want to compare? "))
    places = [input(f"Enter the name of place {i + 1} (State/Province or Country): ").strip() for i in range(n)]

    filtered_data = pd.DataFrame()
    for place in places:
        result = data[(data["State/Province"].str.contains(place, case=False)) |
                       (data["Country"].str.contains(place, case=False))]
        if not result.empty:
            filtered_data = pd.concat([filtered_data, result], ignore_index=True)

    if filtered_data.empty:
        print("No matching places found.")
        return

    columns_to_compare = ["Region", "Country", "State/Province", "Cost of Living (Index)",
                          "Employment Rate (%)", "Crime Severity Index",
                          "Environmental Performance Index", "Average Income (USD)"]

    print("\nComparison of Selected Places:")
    comparison_result = filtered_data[columns_to_compare].transpose()
    print(comparison_result)

    save_to_csv = input("\nDo you want to save this comparison to a CSV file? (yes/no): ").strip().lower()
    if save_to_csv == 'yes':
        output_file = "comparison_result.csv"
        comparison_result.to_csv(output_file)
        print(f"\nComparison saved to '{output_file}'.")

# Regression Analysis Function
def regression_analysis(data):
    target = "Average Income (USD)"
    features = ["Cost of Living (Index)", "Employment Rate (%)",
                "Crime Severity Index", "Environmental Performance Index",
                "GDP (in USD Trillions)"]

    data = data[features + [target]].dropna()
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    print("\nEnter values for prediction:")
    cost_of_living = float(input("Cost of Living Index: "))
    employment_rate = float(input("Employment Rate (%): "))
    crime_severity = float(input("Crime Severity Index: "))
    environment_performance = float(input("Environmental Performance Index: "))
    gdp = float(input("GDP (in USD Trillions): "))

    new_input = [[cost_of_living, employment_rate, crime_severity, environment_performance, gdp]]
    predicted_income = model.predict(new_input)
    print(f"\nPredicted Average Income (USD): {predicted_income[0]:,.2f}")

# Exploratory Data Analysis Function
def exploratory_data_analysis(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

# Cluster Analysis Function
def cluster_analysis(data):
    X = data[["Cost of Living (Index)", "Employment Rate (%)",
               "Crime Severity Index", "Environmental Performance Index"]].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    data['Cluster'] = kmeans.labels_

    sns.scatterplot(data=data, x="Cost of Living (Index)", y="Average Income (USD)",
                    hue="Cluster", palette="viridis")
    plt.title("K-Means Clustering")
    plt.show()

# Principal Component Analysis Function
def principal_component_analysis(data):
    X = data[["Cost of Living (Index)", "Employment Rate (%)",
               "Crime Severity Index", "Environmental Performance Index"]].dropna()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title("PCA - Dimensionality Reduction")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Main menu to run various analyses
def main():
    while True:
        print("\nSelect an option:")
        print("1. Suggest Places")
        print("2. Compare Multiple Places")
        print("3. Regression Analysis")
        print("4. Exploratory Data Analysis")
        print("5. Cluster Analysis")
        print("6. Principal Component Analysis")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")
        if choice == '1':
            suggest_places(data)
        elif choice == '2':
            compare_multiple_places(data)
        elif choice == '3':
            regression_analysis(data)
        elif choice == '4':
            exploratory_data_analysis(data)
        elif choice == '5':
            cluster_analysis(data)
        elif choice == '6':
            principal_component_analysis(data)
        elif choice == '7':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
