
# AC 209A: Alzheimer's Project

# Report EDA

---


```python
# ==================================
# Import libraries
# ==================================

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import pydot
import io

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display

%matplotlib inline

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes',labelsize=16)
```


```python
colors = ["windows blue", "faded green", "greyish", "maroon", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))

c0=sns.color_palette()[0]
c1=sns.color_palette()[1]
c2=sns.color_palette()[2]
c3=sns.color_palette()[3]
c4=sns.color_palette()[4]
```


```python
# ==================================
# Data prep
# ==================================

# Read data
data = pd.read_csv('../data/tadpole/TADPOLE_D1_D2.csv', low_memory=False);
print("Observations in import file: {}".format(len(data)))

# Create 'year' variable
data['YEAR'] = data['EXAMDATE'].apply(lambda x: x[:4])

# Subset data to baseline observations for ADNI 1
data_base = data.loc[(data['VISCODE'] == "bl") & (data['ORIGPROT'] == "ADNI1"), :]
print("Observations in baseline set: {}".format(len(data_base)))

```

    Observations in import file: 12741
    Observations in baseline set: 819



```python
# ===================================================
# Response
# ===================================================
```


```python
# Number of subjects / patients by baseline diagnosis in ADNI 1
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.countplot(x="DX_bl", data=data_base, ax=ax, color=c0, order=['CN', 'LMCI', 'AD']);
ax.set_xlabel("Baseline diagnosis", fontsize=16)
ax.set_ylabel("Number of subjects", fontsize=16)
ax.set_title("Subjects by baseline diagnosis", fontsize=18);

```


![png](Report_EDA_files/Report_EDA_5_0.png)



```python
# Number of subjects / patients by baseline diagnosis in ADNI 1

def get_subjects_by_cohort(data, cohort):
    df = data.copy()
    df = df[df['ORIGPROT'] == cohort]
    df = pd.DataFrame(df.groupby(['YEAR', 'ORIGPROT']).agg({'RID': 'nunique'}))
    df.reset_index(inplace=True)
    df = df[['YEAR', 'RID']]
    
    return df

# Summarize
df1 = get_subjects_by_cohort(data, "ADNI1")
df2 = get_subjects_by_cohort(data, "ADNI2")
df3 = get_subjects_by_cohort(data, "ADNIGO")

# Merge
df = pd.merge(df1, df2, how='left', on="YEAR")
df = pd.merge(df, df3, how='left', on="YEAR")
df.columns = ['YEAR', 'ADNI1', 'ADNI2', 'ADNIGO']
df.fillna(0, inplace=True)
df = df[1:-1]
ind = np.arange(len(df))

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

plt.bar(ind, df['ADNI1'], bottom=df['ADNIGO'], color=c0, label="ADNI1", alpha=0.5)
plt.bar(ind, df['ADNI2'], bottom=df['ADNI1'] + df['ADNIGO'], color=c1, label="ADNI2", alpha=0.5)
plt.bar(ind, df['ADNIGO'], color=c2, label="ADNIGO", alpha=0.5);

plt.xticks(ind, df['YEAR']);
ax.set_xlabel("Year", fontsize=16)
ax.set_ylabel("Number of subjects", fontsize=16)
ax.set_title("ADNI phases over time", fontsize=18)


plt.legend(fontsize=16)

plt.savefig("images\Subjects_by_phase_histogram.png");
```


![png](Report_EDA_files/Report_EDA_6_0.png)



```python
# Number of vistis by visit code 
# (providing some context on longitudinal nature of data)

# Data
g = data.loc[data.ORIGPROT == 'ADNI1', :].groupby('M')['RID'].count()
df = g.to_frame().reset_index()
df = df[df.M < 100]
ind = np.arange(len(df))

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plt.bar(ind, df.RID, color=c0, alpha=0.5)
plt.xticks(ind, df['M']);
ax.set_xlabel("Months since baseline", fontsize=16)
ax.set_ylabel("Visits", fontsize=16)
ax.set_title("Longitudinal observations", fontsize=18);

plt.savefig("images\Visits_over_time.png");

```


![png](Report_EDA_files/Report_EDA_7_0.png)



```python
pd.value_counts(data_base.PTGENDER)
```




    Male      477
    Female    342
    Name: PTGENDER, dtype: int64




```python
#
# Demographics (age, gender, education, marital status)
#


fig, axes = plt.subplots(1, 4, figsize=(24, 6))
ax = axes.ravel()

# Age
ax[0].hist(data_base["AGE"], bins=20, color=c0)
ax[0].set_xlabel("Age")
ax[0].set_ylabel("Subjects")

# Gender
sns.countplot(x="PTGENDER", data=data_base, ax=ax[1], color=c0, saturation=1)
ax[1].set_xlabel("Gender")
ax[1].set_ylabel("")

# Education
ax[2].hist(data_base["PTEDUCAT"], bins=10, color=c0)
ax[2].set_xlabel("Years education")
ax[2].set_ylabel("")

# Gender
df = data_base.loc[data_base.PTMARRY != "Unknown", :]
sns.countplot(x="PTMARRY", data=df, ax=ax[3], color=c0, saturation=1)
ax[3].set_xlabel("Marital status")
ax[3].set_ylabel("");

plt.savefig("images\Demographics_Histograms.png")
```


![png](Report_EDA_files/Report_EDA_9_0.png)



```python
#
# Cognitive tests (Boxplots)
#

cols = ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate"]

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
ax = axes.ravel()

for c in cols:
    i = cols.index(c)
    sns.boxplot(x="DX_bl", y=c, data=data_base, color=c1, ax=ax[i], order=["CN", "LMCI", "AD"], saturation=0.7)
    ax[i].set_ylabel(c)
    ax[i].set_xlabel("")
    
plt.savefig("images\CognitiveTests_Boxplots.png")
```


![png](Report_EDA_files/Report_EDA_10_0.png)



```python
#
# Cognitive tests (scatter plots)
#

cols = ["ADAS11", "MMSE", "RAVLT_immediate"]

g = sns.PairGrid(data_base.sample(100, random_state=97), vars=cols, hue="DX_bl", size=3.5)
g.map_diag(sns.kdeplot)
g.map_offdiag(plt.scatter, s=25)

handles = g._legend_data.values()
labels = g._legend_data.keys()
g.fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14)
g.fig.subplots_adjust(top=0.9, bottom=0.1)

plt.savefig("images\CognitiveTests_Scatterplots.png");
```


![png](Report_EDA_files/Report_EDA_11_0.png)



```python
#
# PET / MRI measures (Boxplots)
#

cols = ["FDG", "Hippocampus", "WholeBrain", "Entorhinal"]

df = data_base.copy()

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
ax = axes.ravel()

for c in cols:
    i = cols.index(c)
    sns.boxplot(x="DX_bl", y=c, data=df, color=c1, ax=ax[i], order=["CN", "LMCI", "AD"], saturation=0.7, fliersize=5)
    ax[i].set_ylabel(c)
    ax[i].set_xlabel("")
    ax[i].set_yticklabels("")
    
plt.savefig("images\PET_MRI_Boxplots.png")
```


![png](Report_EDA_files/Report_EDA_12_0.png)



```python
#
# Metadata summary
#

# Read data
metadata = pd.read_csv("..\data\metadata_new.csv")

df = metadata.groupby('cat').agg({'id': 'count', 'ppn_missing_baseline': 'mean', 'ppn_missing_all': 'mean'})
df.reset_index(inplace=True)
df.ppn_missing_baseline = round(df.ppn_missing_baseline * 100, 1)
df.ppn_missing_all = round(df.ppn_missing_all * 100, 1)

df.columns = ['Category', 'Count', 'Baseline visits', 'All']
```


```python
df['Label'] = df.apply(lambda row: row['Category'].replace("(", "").replace(")", "") + " (" + str(row['Count']) + ")", axis=1)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(df.iloc[:,2:4], yticklabels = list(df.Label), cmap="YlOrBr")
ax.set_title("% Missing observations by category", fontsize=16);

plt.savefig("images\Predictor_Heatmap.png");
```


![png](Report_EDA_files/Report_EDA_14_0.png)



```python
#
# Colinearity of ADAS11 (cognitive test) with other features
#

df = data_base.sample(100, random_state=87)
sns.pairplot(df, x_vars=["FDG", "Hippocampus", "Entorhinal", "MidTemp"], y_vars=["ADAS11"], hue="DX_bl", size=4, kind="reg")
    
plt.savefig("images\ASAS_Coll_Scatterplots.png")
```


![png](Report_EDA_files/Report_EDA_15_0.png)



```python
#Display distribution of different questions on the ADAS13 test

# Read data
data_new = pd.read_csv('../data/data_all.csv', low_memory=False);

# Create 'year' variable
data_new['YEAR'] = data_new['EXAMDATE'].apply(lambda x: x[:4])

# Subset data to baseline observations for ADNI 1
data_new = data_new.loc[(data_new['VISCODE'] == "bl") & (data_new['ORIGPROT'] == "ADNI1"), :]
```


```python
# Plot
cols = [c for c in data_new.columns if c[:6] == "ADAS_Q"]
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
ax = axes.ravel()

for c in cols[:8]:
    i = cols.index(c)
    sns.boxplot(x="DX_bl", y=c, data=data_new, color=c1, ax=ax[i])
    
plt.savefig("images\ASAS_IndividualQs_Boxplots.png")
```


![png](Report_EDA_files/Report_EDA_17_0.png)



```python

```
