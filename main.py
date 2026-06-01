import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print(" DATASET INFO ")
print(df.info())

print(" SUMMARY STATISTICS ")
print(df.describe())

print("MISSING VALUES ")
print(df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("histogram.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.savefig("boxplot.png")
plt.close()

pair_plot = sns.pairplot(
    df[["Age", "Fare", "Pclass", "Survived"]]
)
pair_plot.savefig("pairplot.png")
plt.close()

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("\n===== BASIC INFERENCES =====")

print(f"Average Age: {df['Age'].mean():.2f}")
print(f"Median Age: {df['Age'].median():.2f}")

print(f"Average Fare: {df['Fare'].mean():.2f}")
print(f"Maximum Fare: {df['Fare'].max():.2f}")

if df["Fare"].mean() < df["Fare"].max():
    print("Fare values show the presence of extreme observations.")

if df["Age"].mean() > df["Age"].median():
    print("Age distribution is slightly right-skewed.")
elif df["Age"].mean() < df["Age"].median():
    print("Age distribution is slightly left-skewed.")
else:
    print("Age distribution is approximately symmetric.")

print("\nEDA Completed Successfully!")

print("\nGenerated Files:")
print("- histogram.png")
print("- boxplot.png")
print("- pairplot.png")
print("- correlation_heatmap.png")