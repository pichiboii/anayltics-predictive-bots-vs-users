# Laporan Proyek Machine Learning - Hafizha Aghnia Hasya
**Project Predictive Analytics : Users vs Bots Classification**

## **Domain Project**
Media sosial telah menjadi bagian penting dalam kehidupan digital masyarakat modern, termasuk di Rusia dengan platform VK.com (VKontakte) sebagai salah satu jejaring sosial terbesar. Di balik kemudahan komunikasi dan distribusi informasi yang ditawarkan, VK.com juga menghadapi tantangan serius dalam hal validitas identitas akun pengguna. Meningkatnya jumlah akun palsu atau bot yang beroperasi secara otomatis telah menimbulkan berbagai masalah, mulai dari penyebaran spam, manipulasi opini publik, hingga penyalahgunaan untuk kepentingan politik dan ekonomi.

Dalam konteks spam detection di media sosial, akun yang dioperasikan secara otomatis dapat dibedakan berdasarkan tingkat adopsi otomasi, yakni bot-assisted humans dan human-assisted bots. Yang pertama lebih sulit dibedakan dari akun asli karena perpaduan aktivitas otomatis dan manusia yang lebih halus dan kompleks. Hal ini membuat proses deteksi menjadi lebih menantang, terutama ketika aktivitas bot dirancang agar menyerupai pola perilaku pengguna normal (Deshmukh, 2021).

Masalah ini semakin mendesak untuk diatasi seiring meningkatnya upaya manipulasi informasi publik dan ancaman keamanan digital. Oleh karena itu, proyek ini bertujuan untuk membangun model prediksi yang dapat mengklasifikasikan akun VK.com sebagai bot atau genuine user, menggunakan fitur-fitur numerik dan kategorikal dari profil pengguna. Model prediktif yang efektif tidak hanya dapat membantu VK.com menjaga integritas dan kepercayaan penggunanya, tetapi juga dapat diaplikasikan secara lebih luas pada platform lain yang menghadapi tantangan serupa.


## **Business Understanding**

### Problem Statements
1. Apa model machine learning yang paling efektif untuk mengklasifikasikan akun sebagai bot atau genuine user?
2. Bagaimana performa model machine learning tersebut dalam mengklasifikasi akun sebagai bot atau genuine user?

### Goals
- Membangun model prediksi berbasis machine learning yang mampu mengklasifikasikan akun VK.com sebagai akun bot atau genuine user secara otomatis
- Mencapai performa model yang optimal berdasarkan metrik evaluasi akurasi, precision, recall, F1-score, dan roc-auc
- Mengevaluasi dan membandingkan efektivitas beberapa algoritma dalam konteks deteksi akun bot

### Solution Statements
1.  Membangun model klasifikasi dengan tiga algoritma, yaitu Random Forest, XGBoost, dan K-Nearest Neighbors (KNN), untuk membandingkan performa masing-masing dalam mendeteksi akun bot dengan menggunakan metrik evaluasi yang sudah ditentukan
2. Melakukan hyperparameter tuning pada algoritma tersebut menggunakan grid search

## **Data Understanding**
### Sumber Dataset
https://www.kaggle.com/datasets/juice0lover/users-vs-bots-classification/
### Deskripsi Dataset
- Merupakan data profil publik VK.com, yang mencakup human user yang terverifikasi maupun akun bot yang terverifikasi, dan merepresentasikan kondisi jejaring sosial di dunia nyata dengan profil yang tidak lengkap.
- Terdapat total 2662 data duplikat
- Terdapat 31 fitur yang tidak memberikan banyak informasi (> 75% missing value / unknown) sehingga tidak digunakan / didrop untuk efektivitas modeling
- Terdapat 5 fitur yang redundan karena memiliki korelasi yang tinggi (> 90% korelasi) sehingga tidak digunakan / didrop untuk efektivitas modeling
- Data yang digunakan sebesar 3212 baris data dan 24 fitur

### Deskripsi Fitur yang digunakan
1. has_domain : Apakah pengguna memiliki domain khusus pada profilnya (0 = tidak, 1 = ya, -1 = unknown).
2. has_birth_date : Apakah tanggal lahir tersedia di profil (0 = tidak, 1 = ya, -1 = unknown).
3. has_photo : Apakah pengguna memiliki foto profil (0 = tidak, 1 = ya, -1 = unknown).
4. can_post_on_wall : Apakah pengguna dapat memposting di *wall* (0 = tidak, 1 = ya, -1 = unknown)
5. can_send_message : Apakah pengguna dapat mengirim pesan (0 = tidak, 1 = ya, -1 = unknown).
6. has_website : Apakah pengguna mencantumkan website pribadi (0 = tidak, 1 = ya, -1 = unknown).
7. gender : Jenis kelamin pengguna (1 = female, 2 = male, -1 = unknown).
8. has_last_name : Apakah tersedia nama belakang (0 = tidak, 1 = ya, -1 = unknown).
9. access_to_closed_profile : Apakah pengguna mengakses profil tertutup (0 = tidak, 1 = ya, -1 = unknown).
10. **target** : Label target (user = 0, bot = 1).
11. has_nickname : Apakah pengguna menyertakan nama panggilan (0 = tidak, 1 = ya, -1 = unknown).
12. has_maiden_name : Apakah pengguna menyertakan nama gadis (0 = tidak, 1 = ya, -1 = unknown).
13. has_mobile : Apakah pengguna mencantumkan nomor ponsel (0 = tidak, 1 = ya, -1 = unknown).
14. all_posts_visible : Apakah semua postingan pengguna terlihat publik (0 = tidak, 1 = ya, -1 = unknown).
15. audio_available : Apakah pengguna memiliki file audio di profil (0 = tidak, 1 = ya, -1 = unknown).
16. can_add_as_friend : Apakah pengguna bisa ditambahkan sebagai teman (0 = tidak, 1 = ya, -1 = unknown).
17. can_invite_to_group : Apakah pengguna bisa diundang ke grup (0 = tidak, 1 = ya, -1 = unknown).
18. subscribers_count : Jumlah pengikut akun.
19. is_verified : Apakah akun ini terverifikasi (0 = tidak, 1 = ya, -1 = unknown).
20. has_status : Apakah pengguna memiliki status (bio pendek, dll) (0 = tidak, 1 = ya, -1 = unknown).
21. city : Kota tempat tinggal pengguna.
22. has_occupation : Apakah pengguna menyebutkan pekerjaan (0 = tidak, 1 = ya, -1 = unknown).
23. occupation_type_university : Apakah pekerjaan terkait universitas (akademik/mahasiswa) (0 = tidak, 1 = ya, -1 = unknown).
24. occupation_type_work : Apakah pekerjaan terkait profesional/kerja (0 = tidak, 1 = ya, -1 = unknown).

### *Exploratory Data Analysis*
Exploratory Data Analysis (EDA) adalah tahap eksplorasi data yang telah melalui proses pembersihan untuk memahami karakteristik, distribusi, pola, dan hubungan antar variabel dalam dataset. Pada tahap ini, berbagai teknik analisis statistik dan visualisasi data digunakan untuk mengidentifikasi tren, outlier, dan korelasi yang dapat memberikan wawasan lebih dalam terhadap data. EDA bertujuan untuk membantu menjawab pertanyaan analisis, menemukan pola tersembunyi, serta menjadi dasar dalam pengambilan keputusan sebelum melanjutkan ke tahap pemodelan atau analisis lebih lanjut.

Pada project ini, akan dilakukan descriptive statistics, univariate, dan multivariate analysis.
#### a. *Descriptive Statistics*
Statistik deskriptif bertujuan untuk memberikan gambaran umum mengenai karakteristik data secara numerik. Ukuran statistik seperti mean, median, standar deviasi, nilai minimum, maksimum, serta kuartil digunakan untuk menganalisis fitur numerik dalam dataset.

Dalam project ini, statistik deskriptif digunakan untuk memahami distribusi nilai dari setiap fitur numerik. Salah satu fitur numerik yang dianalisis adalah subscribers_count. Hasil analisis menunjukkan adanya kemungkinan outlier, di mana nilai kuartil ketiga (Q3) berada pada angka 1303.5, sedangkan nilai maksimum mencapai 103729. Perbedaan yang sangat besar antara Q3 dan nilai maksimum ini mengindikasikan adanya penyebaran data yang tidak merata serta potensi keberadaan nilai ekstrem dalam data.

Sementara itu, untuk fitur kategorik seperti city, terdapat 362 nilai unik. Kota dengan frekuensi kemunculan tertinggi adalah Saint Petersburg, yang muncul sebanyak 1.240 kali atau sekitar 38,6% dari total data. Hal ini menunjukkan dominasi kota tersebut dalam distribusi data pada fitur ini.
#### b. *Univariate & Multivariate Analysis*
Univariate analysis bertujuan untuk memahami distribusi data dari satu fitur secara individual, sedangkan multivariate analysis digunakan untuk mengidentifikasi pola dan hubungan antar dua atau lebih fitur dalam dataset.

Pada project ini, analisis univariat menghasilkan visualisasi distribusi data seperti ditunjukkan pada gambar berikut :  
![Distribusi Univariate](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/distribusi%20univariate.png)
Beberapa fitur terlihat didominasi oleh satu nilai. Sebagai contoh, fitur has_domain dan has_last_name mayoritas bernilai 1. Selain itu, distribusi pada fitur subscribers_count menunjukkan pola right-skewed, yang mengindikasikan adanya outlier dengan nilai yang jauh lebih besar dari mayoritas data.

Analisis univariat juga diterapkan pada fitur target, dengan hasil sebagai berikut :
![Distribusi Target](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/distribusi%20target.png)
Dari grafik tersebut terlihat bahwa terdapat class imbalance pada fitur target, di mana mayoritas data memiliki nilai 0 (bukan bot), menunjukkan ketidakseimbangan kelas yang cukup signifikan.

Sementara itu, analisis multivariat memberikan hasil berupa heatmap korelasi antar fitur :
![Heatmap](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/heatmap.png)
Beberapa pasangan fitur yang memiliki korelasi yang cukup tinggi, seperti has_occupation dan occupation_type_university dengan korelasi sebesar 0,89 (89%). Namun demikian, tidak terdapat fitur yang menunjukkan korelasi kuat terhadap fitur target.
    
Selain itu, multivariat analisis juga mengeksplorasi hubungan antara fitur numerik dan fitur target, seperti ditunjukkan pada visualisasi berikut:
![Fitur vs Target](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/fitur%20vs%20target.png)
Dari grafik tersebut terlihat bahwa akun dengan subscribers_count yang lebih tinggi cenderung merupakan akun asli (target = 0). Sementara itu, untuk fitur lainnya tidak ditemukan pola yang signifikan terhadap nilai target.

## **Data Preparation**
Tahap Data Preparation bertujuan untuk menyiapkan data secara optimal sebelum memasuki proses pemodelan. Proses ini sangat penting karena kualitas dan struktur data yang baik akan berdampak langsung pada performa model. Langkah yang dilakukan dalam tahap ini meliputi feature encoding, split data, handling class imbalance, dan feature transformation.
### *Feature Encoding*
Langkah ini dilakukan untuk mengubah fitur kategorik menjadi format numerik agar dapat diproses oleh algoritma machine learning, yang umumnya hanya dapat bekerja dengan data numerik. 

Dalam project ini, metode Label Encoding digunakan untuk mengkonversi label kategorik menjadi angka.
### *Split Data*
Lanngkah ini melakukan pembagian data yang bertujuan untuk menghindari overfitting dan memastikan bahwa evaluasi model dilakukan pada data yang belum pernah dilihat sebelumnya, sehingga performa model dapat diukur secara objektif.

Pada project ini, dataset dibagi menjadi dua bagian: data pelatihan dan data pengujian, dengan rasio 80:20 menggunakan fungsi train_test_split
### *Handling Class Imbalance*
Terdapat ketidakseimbangan jumlah antara kelas target, di mana jumlah akun pengguna (user) jauh lebih banyak dibandingkan akun bot. Ketidakseimbangan kelas dapat menyebabkan model bias terhadap kelas mayoritas. Dengan menyeimbangkan data, model dapat belajar secara lebih adil dan menghasilkan prediksi yang lebih akurat pada kedua kelas.

Untuk mengatasi hal ini, digunakan metode SMOTE (Synthetic Minority Over-sampling Technique) yang melakukan oversampling terhadap kelas minoritas dengan mensintesis data baru.
### *Feature Transformation*
Transformasi dilakukan pada fitur numerik untuk menyesuaikan skala dan distribusinya. Beberapa algoritma machine learning, seperti K-Nearest Neighbors, sensitif terhadap skala fitur. Standarisasi membantu mempercepat konvergensi model dan meningkatkan akurasi, terutama ketika fitur memiliki rentang nilai yang sangat berbeda.

Dalam project ini, digunakan metode StandardScaler untuk melakukan standarisasi, yaitu mengubah data sehingga memiliki mean = 0 dan standar deviasi = 1.
## **Modeling**
Tahap ini bertujuan untuk membangun dan melatih model klasifikasi guna membedakan antara akun bot dan akun user berdasarkan fitur-fitur profil serta aktivitas pengguna. Tahapan proses pemodelan dimulai dengan inisialisasi model, pelatihan model / train model, predict model, dan evaluasi model.

Tiga algoritma yang digunakan dalam proyek ini adalah Random Forest, XGBoost, dan K-Nearest Neighbors (KNN). Ketiga model dilatih menggunakan parameter default sebagai berikut:
```
rf = RandomForestClassifier()
xgb = XGBClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
```
Evaluasi performa model dilakukan menggunakan lima metrik yaitu accuracy, precision, recall, F1-score, dan ROC-AUC, dengan hasil sebagai berikut :
![Evaluasi 1](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/Screenshot%202025-05-24%20113708.png)
Berdasarkan hasil evaluasi tersebut, model XGBoost menunjukkan performa terbaik secara konsisten di hampir semua metrik, diikuti oleh Random Forest, dan terakhir KNN.

### Kelebihan dan Kekurangan Masing-Masing Algoritma:
#### *Random Forest*
✅ Kelebihan:
- Tahan terhadap overfitting karena menggunakan banyak pohon.
- Dapat menangani data numerik dan kategorikal dengan baik.
- Memiliki built-in fitur penting (feature importance).

❌ Kekurangan:
- Kurang optimal jika data tidak dituning dengan baik.
- Cenderung lebih lambat untuk prediksi dibanding model sederhana.

#### *XGBoost*
✅ Kelebihan:
- Performa tinggi dan efisien, terutama pada data tabular.
- Mampu menangani missing value secara internal.
- Bisa menangani class imbalance dengan parameter khusus.

❌ Kekurangan:
- Membutuhkan tuning hyperparameter agar mencapai performa maksimal.
- Proses training lebih kompleks dibanding Random Forest.

#### *K-Nearest Neighbors (KNN)*
✅ Kelebihan:
- Algoritma sederhana dan mudah diimplementasikan.
- Tidak membutuhkan proses training (lazy learner).

❌ Kekurangan:
- Sensitif terhadap skala dan noise pada data.
- Tidak cocok untuk dataset besar karena membutuhkan banyak waktu untuk prediksi.
- Kinerja menurun jika fitur tidak ditransformasi dengan baik.


### *Hyperparameter Tuning*
Hyperparameter tuning dilakukan sebagai bagian dari proses model improvement, yaitu untuk meningkatkan kinerja model di atas baseline yang sebelumnya dibangun menggunakan parameter default. Hyperparameter adalah parameter yang ditetapkan sebelum proses pelatihan dimulai, dan memiliki pengaruh langsung terhadap kompleksitas model, performa prediksi, serta risiko overfitting atau underfitting.

Pada tahap ini, dilakukan pencarian kombinasi hyperparameter terbaik menggunakan metode Grid Search, yang mengevaluasi seluruh kemungkinan kombinasi parameter pada ruang pencarian (search space) yang telah ditentukan. Proses ini dikombinasikan dengan teknik cross-validation dan selama proses validasi, skor evaluasi yang digunakan untuk memilih parameter terbaik adalah F1-score.

Berikut adalah ruang pencarian hyperparameter yang digunakan untuk masing-masing model:
```
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
```
Setelah dilakukan GridSearch, didapatkan parameter terbaik untuk ketiga model tersebut adalah :
```
Random Forest : {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300}
XGBoost : {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100} 
KNN : {'n_neighbors': 3, 'weights': 'distance'}
```

## *Evaluation*
Tahap evaluasi bertujuan untuk mengukur dan membandingkan performa model klasifikasi dalam membedakan antara akun bot dan akun user. Evaluasi dilakukan menggunakan lima metrik utama, yaitu Accuracy, Precision, Recall, F1-Score, dan ROC-AUC Score. Masing-masing metrik memberikan sudut pandang yang berbeda terkait kemampuan model dalam menangani prediksi, terutama pada dataset dengan ketidakseimbangan kelas.
### Metrik Evaluasi yang Digunakan
#### 1. *Accuracy*
Mengukur proporsi prediksi yang benar dibandingkan dengan total seluruh prediksi. Formula accuracy adalah sebagai berikut :

![Accuracy Formula](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/accuracy%20formula.png)

Cocok digunakan ketika distribusi kelas seimbang, namun bisa menyesatkan pada dataset dengan class imbalance.
#### 2. *Precision*
Mengukur seberapa akurat model dalam memprediksi kelas positif (dalam kasus ini, akun bot). Formula precision adalah sebagai berikut :

![Precision Formula](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/precision%20formula.png)

Tinggi jika model jarang salah memberi label positif. Penting saat false positive perlu diminimalkan.
#### 3. *Recall*
Mengukur kemampuan model dalam menemukan semua contoh kelas positif yang sebenarnya. Formula recall adalah sebagai berikut :

![Recall Formula](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/recall%20formula.png)

Tinggi jika model jarang melewatkan akun bot. Penting saat false negative sangat merugikan.
#### 4. *F1-Score*
Merupakan rata-rata harmonis dari precision dan recall, yang memberikan keseimbangan antara keduanya. Formula f1-score adalah sebagai berikut :

![F1-Score Formula](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/f1-score%20formula.png)

Cocok digunakan pada kasus class imbalance untuk mendapatkan gambaran performa model secara menyeluruh
#### 5. ROC-AUC Score (Receiver Operating Characteristic - Area Under Curve)
Mengukur kemampuan model dalam membedakan antara kelas positif dan negatif pada berbagai threshold. Nilainya berkisar antara 0–1, semakin mendekati 1 maka semakin baik.
- ROC Curve menggambarkan trade-off antara true positive rate (recall) dan false positive rate.
- AUC menunjukkan seberapa besar kemungkinan model memeringkat sampel positif lebih tinggi daripada negatif.

### *Model Evaluation*
Setelah dilakukan hyperparameter tuning menggunakan GridSearchCV dengan metrik utama f1_score, ketiga model — Random Forest, XGBoost, dan K-Nearest Neighbors (KNN) — menunjukkan peningkatan performa.
![Evaluasi Tuned](https://github.com/pichiboii/anayltics-predictive-bots-vs-users/blob/main/archive/Screenshot%202025-05-24%20115040.png)
XGBoost memperoleh skor tertinggi di semua metrik, terutama pada precision (0.9042), f1-score (0.9015), dan ROC AUC (0.9326). Ini menunjukkan bahwa XGBoost tidak hanya mampu mengklasifikasikan dengan akurasi tinggi, tetapi juga secara seimbang dalam mendeteksi bot dan user (precision vs recall).

Random Forest memiliki performa yang sangat mirip dengan XGBoost, hanya sedikit lebih rendah terutama pada precision dan f1-score. Model ini tetap kompetitif dengan kelebihan berupa kestabilan dan interpretabilitas.

KNN berada di posisi terakhir, dengan skor precision (0.7710) dan f1-score (0.7954) yang lebih rendah dibandingkan dua model lainnya. Hal ini menunjukkan bahwa model KNN kurang efektif dalam membedakan antara akun bot dan user dalam konteks dataset ini, kemungkinan karena sensitivitas terhadap skala dan noise data.

## **Kesimpulan**
Berdasarkan evaluasi menyeluruh, model **XGBoost menjadi pilihan terbaik** untuk klasifikasi akun bot dan user, mengingat kemampuannya dalam mencapai keseimbangan antara akurasi tinggi dan generalisasi yang baik.

## **Referensi**
Deshmukh, R. (2021). Performance comparison for spam detection in social media using deep learning algorithms. Turkish Journal of Computer and Mathematics Education, 12(1S), 193–201.
