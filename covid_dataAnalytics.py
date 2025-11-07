
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("countrywise_corona.csv")

print("Dataset Loaded", df.shape)
df.head()

classification = {}
for col in df.columns:
    if df[col].dtype == 'object':
        classification[col] = 'Nominal (Categorical)'
    else:
        classification[col] = 'Ratio (Numeric Count or Rate)'
pd.DataFrame.from_dict(classification, orient='index', columns=['Data Type'])

region_col = 'WHO Region' if 'WHO Region' in df.columns else df.columns[1]
for col in ['Confirmed', 'Deaths']:
    print(f"\nSummary statistics for {col} by {region_col}")
    print(df.groupby(region_col)[col].agg(['count','mean','median','std','min','max']).round(2))

print("\ndata cleaning")

print("Missing values before:\n", df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[num_cols] = df[num_cols].abs()
df = df.drop_duplicates()
df[num_cols] = df[num_cols].fillna(0)
print("Missing values after:\n", df.isnull().sum())

for col in ['Confirmed', 'Deaths']:
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"{col} Histogram")
    plt.subplot(1,2,2)
    sns.boxplot(x=df[col])
    plt.title(f"{col} Boxplot")
    plt.tight_layout()
    plt.show()

col = 'Confirmed'
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = df[(df[col] < lower) | (df[col] > upper)]
print(f"\n{col} Outliers count:", outliers.shape[0])
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.boxplot(x=df[col]); plt.title("Before Outlier Removal")
plt.subplot(1,2,2)
sns.boxplot(x=df[(df[col] >= lower) & (df[col] <= upper)][col])
plt.title("After Outlier Removal")
plt.show()

sm.qqplot(df['Deaths'], line='45', fit=True)
plt.title("Q–Q Plot for Deaths")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
corr_with_deaths = df.corr(numeric_only=True)['Deaths'].sort_values(ascending=False)
print("\nCorrelation with Deaths:\n", corr_with_deaths)

cols = [c for c in df.columns if 'Confirmed' in c or c == 'Deaths']
cols = [c for c in cols if c in df.columns][:3]
sns.pairplot(df, vars=cols, hue=region_col)
plt.suptitle("Pairplot by WHO Region", y=1.02)
plt.show()

regions = df[region_col].dropna().unique()
if len(regions) >= 2:
    r1, r2 = regions[:2]
    grp1 = df[df[region_col]==r1]['Deaths']
    grp2 = df[df[region_col]==r2]['Deaths']
    tstat, pval = stats.ttest_ind(grp1, grp2, equal_var=False, nan_policy='omit')
    print(f"\nT-Test between {r1} and {r2} on Deaths:")
    print(f"T-Statistic: {tstat:.3f}, P-Value: {pval:.4f}")
else:
    print("Not enough regions for t-test")
mean_deaths = df['Deaths'].mean()
std_deaths = df['Deaths'].std()
n = df['Deaths'].count()
se = std_deaths/np.sqrt(n)
moe = stats.t.ppf(0.975, n-1)*se
print(f"\nMean Deaths = {mean_deaths:.2f} ± {moe:.2f} (95% CI)")

if len(regions) > 0:
    r = regions[0]
    data = df[df[region_col]==r]['Confirmed']
    if len(data)>1:
        ci = stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))
        print(f"95% CI for mean Confirmed in {r}: {ci}")
    else:
        print(f"Not enough data for CI in region: {r}")

df_model = df.copy()
df_model = df_model.drop(columns=['Country/Region'], errors='ignore')

region_col = 'WHO Region'
if region_col in df_model.columns:
    df_model = pd.get_dummies(df_model, columns=[region_col], drop_first=True)

y = df_model['Deaths']
X = df_model.drop(columns=['Deaths'])

X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X = X.clip(lower=-1e9, upper=1e9)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nRegression Performance:\nMSE={mse:.2f}\nRMSE={rmse:.2f}\nR²={r2:.4f}")

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Actual vs Predicted Deaths (Linear Regression)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
