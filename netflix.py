import pandas as pd
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Read a CSV file containing Netflix users data
df = pd.read_csv("netflix_users_cleaned.csv")

# Print the first 10 rows of the data
print(df.head(10))

# Print the shape of the data (number of rows and columns)
print("Data Shape:", df.shape)

# Print the number of missing values in each column
print("Missing Values:\n", df.isnull().sum())

# Convert 'Last_Login' column to datetime format and handle errors
df['Last_Login'] = pd.to_datetime(df['Last_Login'], errors='coerce')

# Handle missing values in categorical columns
df['Country'].fillna('Unknown', inplace=True)
df['Favorite_Genre'].fillna('Unknown', inplace=True)

# Check for Age (should be between 13 and 90 years)
df = df[(df['Age'] >= 13) & (df['Age'] <= 90)]

# Check that Watch Time in hours is greater than or equal to 0
df = df[df['Watch_Time_Hours'] >= 0]

# Check for duplicates in the data and remove them
df = df.drop_duplicates()

# Remove extra spaces from user names
df['Name'] = df['Name'].str.strip()

# Save cleaned data
df.to_csv('netflix_users_cleaned.csv', index=False)
print("Data cleaned and saved to netflix_users_cleaned.csv")

# Basic statistics
print(f"The statistics of Age:\n{df['Age'].describe()}")
print(f"The statistics of Watch_Time_Hours:\n{df['Watch_Time_Hours'].describe()}")

# Count users by country, subscription type, and favorite genre
country_counts = df['Country'].value_counts()
print(f"The number of users from each country:\n{country_counts}")

subscription_counts = df['Subscription_Type'].value_counts()
print(f"The number of users from each subscription type:\n{subscription_counts}")

genre_counts = df['Favorite_Genre'].value_counts()
print(f"The number of users from each genre:\n{genre_counts}")

# Correlation between Age and Watch_Time_Hours
correlation = df[['Age', 'Watch_Time_Hours']].corr()
print(f"The correlation between Age and Watch_Time_Hours:\n{correlation}")

# Standard Deviation of Watch Time by Country
watch_time_std = df.groupby('Country')['Watch_Time_Hours'].std()
print("Standard Deviation of Watch Time by Country:\n", watch_time_std)

# Plot age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')

# Plot favorite genre distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Favorite_Genre', data=df)
plt.title('Favorite Genre Distribution')
plt.xlabel('Favorite Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('genre_distribution.png')

# Plot relationship between age and watch time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Watch_Time_Hours', hue='Subscription_Type', data=df)
plt.title('Relationship between Age and Watch Time')
plt.xlabel('Age')
plt.ylabel('Watch Time (Hours)')
plt.savefig('age_vs_watch_time.png')