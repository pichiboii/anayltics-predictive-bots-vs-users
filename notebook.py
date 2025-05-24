#!/usr/bin/env python
# coding: utf-8

# # Project Predictive Analytics : Users vs Bots Classification
# - **Nama:** Hafizha Aghnia Hasya
# - **Email:** mc006d5x2114@student.devacademy.id
# - **ID Dicoding:** MC006D5X2114

# ## Import Library
Pada tahap ini, dilakukan import berbagai library yang diperlukan untuk proses analisis, pemrosesan data, dan pembuatan model klasifikasi.
# In[1]:


import os
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# ## Data Loading
Pada tahap ini, dataset didownload dari Kaggle kemudian dimuat menjadi csv dan diubah menjadi dataframe menggunakan pandas.
# In[2]:


# download dataset dari kaggle

kaggle_json_path = "kaggle.json"

kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

shutil.move(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))

kaggle_json_file = os.path.join(kaggle_dir, "kaggle.json")
try:
    os.chmod(kaggle_json_file, stat.S_IRUSR | stat.S_IWUSR)
except Exception as e:
    print("Permission warning (ignore on Windows):", e)


# In[3]:


get_ipython().system('kaggle datasets download -d juice0lover/users-vs-bots-classification')

# fungsi untuk ekstrak file zip
def unzip_file(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted: {zip_path}")

# ekstrak dataset
unzip_file("users-vs-bots-classification.zip")


# In[4]:


# hapus file zip
if os.path.exists("users-vs-bots-classification.zip"):
    os.remove("users-vs-bots-classification.zip")


# In[5]:


df = pd.read_csv('bots_vs_users.csv')
df.head()


# ## Data Cleaning
Pada tahap ini, akan dilakukan pengecekan tipe data, data duplikat, missing value (berupa null dan 'Unknown'), dan multicolinearity (fitur dengan korelasi tinggi). Kemudian melakukan handling dengan menghapus kolom dengan missing value di atas 75%, mengganti nilai 'Unknown', menyesuaikan tipe data, menghapus duplikat, dan menghapus fitur dengan korelasi tinggi di atas 90%
# In[6]:


df.info()


# In[7]:


# cek duplikat
print(f'Banyak data yang duplikat : {df.duplicated().sum()} \n')

# cek missing value
missing_value_percent = (df.isnull().sum() / len(df)) * 100
print('Data yang memiliki missing value : ')
print(missing_value_percent[missing_value_percent > 0], '\n')

# cek data dengan value 'Unknown'
unknown_percent = (df.apply(lambda col: col.str.lower().eq('unknown').sum() if col.dtypes == 'object' else 0) / len(df)) * 100
print('Jumlah nilai "unknown" di setiap kolom:')
print(unknown_percent[unknown_percent > 0])


# In[8]:


# hapus kolom yang tidak banyak memberikan informasi
# kolom-kolom yang memiliki missing value atau unknown lebih dari 75%
cols_to_drop = missing_value_percent[missing_value_percent > 75].index.union(
    unknown_percent[unknown_percent > 75].index
)
# drop kolom dari DataFrame
df = df.drop(columns=cols_to_drop)

print(f'Kolom yang di-drop (>%75 missing/unknown):')
for col in cols_to_drop:
    print(f'- {col}')


# In[9]:


'''
nilai unknown pada dataset terdapat pada fitur numerik yang merupakan data nominal, di mana nilai hanya berupa label,
kecuali fitur 'city' yang berupa kategorik.
oleh karena itu, 'unknown' akan diisi dengan -1 untuk membedakannya dengan 0 (false)
'''
# mengganti nilai unknown yang tersisa dengan -1 (kecuali fitur city)
df = df.apply(lambda col: col.str.lower().replace('unknown', -1) if col.dtypes == 'object' and col.name != 'city' else col)


# In[10]:


df.info()


# In[11]:


# konversi tipe data yang seharusnya integer

# list kolom yang ingin dikonversi ke integer (semua kecuali 'city')
cols_to_convert = df.columns.difference(['city'])
# konversi kolom-kolom tersebut ke integer
df[cols_to_convert] = df[cols_to_convert].astype(float)


# In[12]:


# hapus duplikat
df = df.drop_duplicates()


# In[13]:


# inisialisasi fitur numerik
nums = df.select_dtypes(include=np.number).columns

# hitung korelasi antar fitur numerik
corr_matrix = df[nums].corr().abs()  # pakai abs supaya korelasi negatif juga dihitung
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# fitur yang memiliki korelasi > 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print("Kolom yang akan di-drop karena korelasi tinggi:")
print(to_drop)

# drop fitur dari dataframe
df.drop(columns=to_drop, inplace=True)


# In[14]:


# cek duplikat
print(f'Banyak data yang duplikat (after) : {df.duplicated().sum()} \n')

# cek missing value
missing_value_percent_after = (df.isnull().sum() / len(df)) * 100
print('Data yang memiliki missing value (after) : ')
print(missing_value_percent_after[missing_value_percent_after > 0], '\n')

# cek data dengan value 'Unknown'
unknown_percent_after = (df.apply(lambda col: col.str.lower().eq('unknown').sum() if col.dtypes == 'object' else 0) / len(df)) * 100
print('Jumlah nilai "unknown" di setiap kolom (after):')
print(unknown_percent_after[unknown_percent_after > 0], '\n')

# cek fitur dengan korelasi tinggi
nums = df.select_dtypes(include=np.number).columns
corr_matrix = df[nums].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
multicolinearity_features = [column for column in upper.columns if any(upper[column] > 0.9)]
print("Fitur dengan korelasi tinggi:")
print(multicolinearity_features)


# In[15]:


df.info()


# ## Exploratory Data Analysis
Exploratory Data Analysis (EDA) adalah tahap eksplorasi data yang telah melalui proses pembersihan untuk memahami karakteristik, distribusi, pola, dan hubungan antar variabel dalam dataset. Pada tahap ini, berbagai teknik analisis statistik dan visualisasi data digunakan untuk mengidentifikasi tren, outlier, dan korelasi yang dapat memberikan wawasan lebih dalam terhadap data. EDA bertujuan untuk membantu menjawab pertanyaan analisis, menemukan pola tersembunyi, serta menjadi dasar dalam pengambilan keputusan sebelum melanjutkan ke tahap pemodelan atau analisis lebih lanjut.

Pada tahap ini, dilakukan :
- Statistics Descriptive Analysis
- Univariate & Multivariate Analysis
# ### Statistics Descriptive
Tahapan ini bertujuan untuk memperoleh gambaran umum karakteristik data secara numerik. Statistik deskriptif seperti mean, median, standard deviation, minimum, maksimum, dan kuartil digunakan untuk mengetahui sebaran fitur numerik. Selain itu statistik deskriptif juga dapat memberikan gambaran umum pada fitur kategorik.
# In[16]:


# melihat tipe data
df.info()


# In[17]:


# fitur numerik
df.describe().T


# In[18]:


# fitur kategorik
df.describe(exclude=np.number)


# ### Univariate & Multivariate Analysis
Univariate analysis bertujuan untuk memahami distribusi data dari satu fitur secara individual, sedangkan multivariate analysis digunakan untuk mengidentifikasi pola dan hubungan antar dua atau lebih fitur dalam dataset. 
Tahap ini dimulai dari visualisasi distribusi fitur numerik, distribusi fitur kategorik, distribusi fitur target, heatmap untuk melihat korelasi antar fitur, dan scatter plot antara fitur numerik dan fitur target.
# In[19]:


# inisialisasi fitur numerik
nums = df.select_dtypes(include=np.number).columns


# In[20]:


# menampilkan distribusi fitur numerik
fig, axes = plt.subplots(figsize=(15, 10))
fig.suptitle('Distribusi Fitur Numerik', fontsize=16)
for i in range(0, len(nums)):
    plt.subplot(5, 6, i+1)
    sns.distplot(df[nums[i]],color='salmon')
    plt.tight_layout()


# In[21]:


# menampilkan distribusi fitur kategorik (city)

city_counts = df['city'].value_counts()

# ambil 10 kota terbanyak
top10 = city_counts.head(10)

plt.figure(figsize=(10, 6))
top10.plot(kind='bar', color='salmon')
plt.title('10 Kota Terbanyak dalam Data')
plt.ylabel('Jumlah Akun')
plt.xlabel('Kota')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[22]:


# distribusi fitur target
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=df, color='salmon')
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()


# In[23]:


# menampilkan korelasi antar fitur numerik menggunakan heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(df[nums].corr(), cmap='flare', annot=True)
plt.title('Heatmap antar Fitur Numerik', fontsize=16)
plt.show()


# In[24]:


# distribusi fitur numerik terhadap target

nums_no_target = [f for f in nums if f != 'target']

n_cols = 4
n_rows = math.ceil(len(nums_no_target) / n_cols)
plt.figure(figsize=(n_cols * 5, n_rows * 4))

for i, feature in enumerate(nums_no_target):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(df.index[df['target'] == 0], df[df['target'] == 0][feature],
                color='blue', alpha=0.5, label='Target 0')
    plt.scatter(df.index[df['target'] == 1], df[df['target'] == 1][feature],
                color='red', alpha=0.5, label='Target 1')
    plt.title(feature)
    plt.xlabel('Index')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()


# ## Data Preparation
Bertujuan untuk menyiapkan data agar dapat digunakan secara optimal dalam proses pemodelan. Tahapan ini meliputi encoding, split data, mengatasi class imbalance, dan transformasi fitur. 
# ### Feature Encoding
Mengubah data kategorikal atau teks menjadi format numerik agar dapat digunakan oleh model machine learning. Akan digunakan metode label encoding
# In[25]:


# mengubah data kategrorik menjadi numerik menggunakan label encoding

label_encoder = LabelEncoder()
# melakukan encoding untuk fitur city
df['city'] = label_encoder.fit_transform(df['city'])
df.head(2)


# ### Split Data
Memisahkan dataset menjadi data pelatihan dan data pengujian untuk mengevaluasi performa model secara adil dan mencegah overfitting. Akan dilakukan split data 80:20
# In[26]:


# Split Data Training dan Testing
X = df.drop(['target'], axis = 1)
y = df['target']

# Split data 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[27]:


print(X_train.shape)
print(X_test.shape)


# ### Handling Class Imbalance
Menangani ketidakseimbangan jumlah antara kelas target (jumlah akun user jauh lebih banyak daripada akun bot), yang dapat menyebabkan model bias terhadap kelas mayoritas. Akan digunakan metode SMOTE oversampling
# In[28]:


# Initialize SMOTE
smote = SMOTE(random_state=17)

# Doing oversampling to Data Train
X_train_o, y_train_o = smote.fit_resample(X_train, y_train)


# In[29]:


print(X_train_o.shape)


# ### Feature Transformation
Mengubah skala, distribusi, atau format fitur numerik agar sesuai dengan asumsi model dan meningkatkan performa model machine learning. Akan digunakan standarisasi dengan Standard Scaler
# In[30]:


# standarisasi data
scaler = StandardScaler()
X_train_o = scaler.fit_transform(X_train_o)
X_test = scaler.transform(X_test)


# ## Modeling
Membangun dan melatih model klasifikasi untuk membedakan antara akun bot dan akun user berdasarkan fitur-fitur profil dan aktivitas. Terdapat 3 model yang akan dibangun yaitu Random Forest, XGBoost, dan K-Nearest Neighbor
# In[31]:


rf = RandomForestClassifier()
xgb = XGBClassifier()
knn = KNeighborsClassifier(n_neighbors=5)

models = [rf, xgb, knn]
results = {}

for model in models:
    model_name = type(model).__name__
    model.fit(X_train_o, y_train_o)
    y_pred = model.predict(X_test)
    # simpan hasil evaluasi dalam dictionary
    results[model_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    print(f'{model_name} Finished!')


# In[32]:


results_df = pd.DataFrame(results).T  # transpose biar model di baris
display(results_df)


# ### Hyperparameter Tuning
Meningkatkan performa model dengan mencari kombinasi parameter terbaik yang tidak bisa dipelajari secara otomatis selama pelatihan menggunakan GridSearch
# In[33]:


# parameter grid untuk setiap model
param_grids = {
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    },
    'XGBClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2, 0.3]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
}

# inisialisasi ulang model untuk GridSearchCV
models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'XGBClassifier': XGBClassifier(eval_metric='logloss'),
    'KNeighborsClassifier': KNeighborsClassifier()
}

# simpan best model dan hasil tuning
best_models = {}
tuned_results = {}

for name, model in models.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_o, y_train_o)

    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    
    # simpan hasil evaluasi model
    tuned_results[name] = {
        'best_params': grid.best_params_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    
    print(f"{name} tuning finished. Best params: {grid.best_params_} \n")


# ## Model Evaluation
Mengukur dan membandingkan performa model klasifikasi dalam membedakan antara akun bot dan user menggunakan berbagai metrik evaluasi. Pada project ini, menggunakan metrik akurasi, presisi, recall, f1 score, dan roc_auc score
# In[34]:


pd.DataFrame(tuned_results).T


# In[35]:


best_models


# In[ ]:




