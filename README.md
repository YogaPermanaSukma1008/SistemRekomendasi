# Sistem Rekomendasi Film dengan Menggunakan Content Based Filtering dan Collaborative Based Filtering

## Project Overview
Industri kreatif film menunjukkan pertumbuhan signifikan seiring dengan meningkatnya konsumsi media digital di era modern. Digitalisasi turut mendorong transformasi besar dalam distribusi film melalui platform streaming berbasis website seperti Netflix, Disney+, dan lainnya. Kemudahan akses ini menciptakan tantangan baru berupa information overload, di mana pengguna kesulitan memilih film yang sesuai dengan preferensi mereka.
Menurut laporan Netflix, sistem rekomendasi mereka berkontribusi langsung terhadap penghematan lebih dari $1 miliar per tahun dengan mengurangi tingkat churn pelanggan, serta menjaga tingkat retensi pengguna pada angka mengesankan sekitar 2,3% (Gomez-Uribe & Hunt, 2015). Oleh karena itu, sistem rekomendasi berbasis content-based filtering menjadi solusi yang relevan untuk mempersonalisasi pengalaman pengguna dan meningkatkan kepuasan mereka dalam menjelajahi katalog film.


### Business Understanding
Sebuah platform distribusi film digital baru yang diluncurkan di Indonesia beberapa bulan terakhir mengalami tantangan serius dalam mempertahankan jumlah pelanggannya. Setelah awal peluncuran yang menjanjikan, data menunjukkan terjadinya penurunan jumlah subscriber secara bertahap. Berdasarkan analisis ulasan dan umpan balik pengguna, salah satu masalah utama yang muncul adalah kebingungan dalam memilih film untuk ditonton. Meski koleksi film yang tersedia di platform sangat beragam dan banyak, justru hal tersebut menciptakan beban kognitif bagi pengguna dalam membuat keputusan.
Fenomena ini dikenal sebagai paradox of choice—di mana semakin banyak pilihan yang tersedia, semakin besar pula kebingungan yang dialami pengguna. Konsep ini diperkenalkan oleh psikolog Barry Schwartz dalam bukunya "The Paradox of Choice: Why More Is Less", yang menjelaskan bahwa terlalu banyak pilihan dapat menyebabkan kecemasan, penundaan keputusan, bahkan penyesalan setelah memilih. Dalam konteks platform film digital, hal ini berpotensi menurunkan kepuasan pengguna dan meningkatkan risiko churn (berhentinya pelanggan).
Masalah ini semakin diperparah dengan tidaknya tersedia sistem rekomendasi yang personal. Saat ini, film yang ditampilkan di halaman utama cenderung didasarkan pada popularitas umum, bukan preferensi individual pengguna. Akibatnya, pengalaman menonton menjadi kurang relevan dan kurang menarik, serta tidak mendorong eksplorasi konten yang lebih luas. Sebagai pembanding, platform seperti Netflix menyatakan bahwa sistem rekomendasi yang baik mampu menghemat lebih dari $1 miliar setiap tahun dengan cara mengurangi tingkat churn melalui peningkatan pengalaman pengguna (Qin, 2024).

### Problem Statement
Dari studi kasus ini, pernyataan masalah dapat dirumuskan sebagai berikut: 
1.	Pengguna kesulitan menemukan film yang sesuai dengan preferensinya akibat banyaknya pilihan film yang tersedia di platform streaming. 
2.	Sistem rekomendasi yang telah dibangun ternyata tidak sepenuhnya tepat merekomendasikan film yang sesuai dengan preferensi pengguna. 

### Goals
1.	Terbentuk sistem rekomendasi berbasis content based filtering yang dapat merekomendasikan film yang sesuai dengan minat genre film pengguna. 
2.	Meningkatkan akurasi dan relevansi sistem rekomendasi film dengan menerapkan pendekatan atau metode yang lebih efektif dalam memahami preferensi pengguna.

### Solution Statement:
1.	Untuk mengatasi masalah 1, maka dibangun sistem rekomendasi film dengan pendekatan content based filtering. Sistem ini menggunakan TF-IDF Vectorizer pada fitur genres untuk menghitung kemiripan antar film dengan Cosine Similarity.
2.	Jika pendekatan content based filtering menghasilkan rekomendasi yang kurang baik maka akan digunakan sistem rekomendasi dengan pendekatan collaborative filtering dengan beberapa metode seperti SVD atau KNN. 

## Data Understanding
Dataset yang digunakan berasal dari (Kaggle)[ https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system/data]. Dataset ini berisi dua dataset yaitu dataset Movies dan Ratings. 
Pada dataset Movies terdapat beberapa fitur yaitu: 
1.	movieId : kode nomor identitas movie
2.	title : berisi judul movie 
3.	genre : berisi genre pada movie 
Sementara itu, pada dataset rating terdapat beberapa fitur yaitu: 
1.	UserId: kode nomor identitas pengguna
2.	Movieid : kode nomor identitas movie
3.	Rating : tingkat kepuasan pengguna yang dijelaskan menggunakan data ordinal (0,5 – 5)
4.	Timestamps : urutan karakter atau informasi yang menunjukkan kapan suatu peristiwa terjadi. 
Berdasarkan hasil dari data info, dataset movies berjumlah 62423 baris dengan 3 kolom, sedangkan dataset rating berjumlah 25000095 baris dengan 4 kolom.
Selain pemahaman akan data, pada proses ini dilakukan pengecekan data dari data missing dan duplikat. Berdasarkan hasil proses data wrangling tidak ditemukan adanya missing data dan duplikat data pada dataset sehingga tidak diperlukan proses penanganan data hilang dan duplikat. Proses data cleaning dilakukan dengan membersihkan kolom nama film dari value yang bukan angka atau huruf sehingga data siap digunakan pada proses EDA. 

### EDA 
EDA (Exploratory Data Analysis) adalah proses eksplorasi awal terhadap data sebelum modeling, dengan tujuan memahami struktur, pola, anomali, dan hubungan antar variabel dalam dataset. Dalam hal ini, EDA yang digunakan yaitu:
1.	Menampilkan unique value pada kolom genre. 
2.	Menghapus value no genre listed karena tidak relevan. 
3.	Menampilkan film dengan penonton terbanyak (populer) dengan bar chart
4.	Menampilkan jumlah film berdasarkan genre dengan bar chart. 
5.	Penggabungan dataset untuk melihat film dengan rating tertinggi dan rating terendah.

Berdasarkan proses EDA, unik value pada kolom genre berjumlah 20, satu diantaranya tertulis no listed genre. Oleh karena itu diperlukan pengananan berupa penghapusan baris yang memiliki data no listed genre pada kolom genre. Proses ini dilakukan agar film yang direkomendasikan sesuai dengan genre yang diminati pengguna. Adanya data no listed genre akan sangat membingungkan sehingga perlu dihapus. Berdasarkan tahapan nomor 3 ditemukan bahwa film berjudul forest gump merupakan film dengan penonton terbanyak hingga 80.000 rating yang diberikan. Sementara itu genre paling banyak adalah genre drama dengan lebih dari 25000 judul film. Penggabungan dataset diperlukan untuk melihat film dengan raating tertinggi dan terendah karena untuk menampilkan bar chart yang demikian diperlukan kolom tittle pada dataset movies dan kolom rating pada dataset ratings. Ditemukan bahwa rata - rata tertinggi pada film mencapai 5 sementara terendah adalah 0.5. 
   

### Data Preparation
#### Data Cleaning
Data cleaning merupakan pembersihan data. Pembersihan data dalam proyek ini adalah pembersihan pada kolom genre dengan menghapus tanda pemisah (|) untuk memudahkan dalam proses vektorisasi TF-IDF. 
#### Data Sampling
Proyek ini menggunakan data sampling sejumlah 10.000 dikarenakan adanya keterbatasan RAM pada google collab. Metode yang digunakan untuk pengambilan data sampel adalah random sampling dengan parameter random_state = 42 untuk memastikan bahwa eksperimen atau analisis dapat diulang dengan hasil yang sama, sehingga memudahkan untuk membandingkan performa model atau metode yang berbeda.
#### Vektorisasi (Content Based Filtering)
Vektorisasi digunakan untuk mengubah data genre film (teks) menjadi bentuk numerik yang dapat dipahami oleh model machine learning. Metode yan digunakan adalah TF-IDF. Metode ini TF-IDF menilai seberapa penting sebuah kata dalam suatu dokumen relatif terhadap seluruh koleksi dokumen (corpus). Representasi numerik  nantinya akan digunakan dalam sistem rekomendasi film untuk menghitung kesamaan antara film-film berdasarkan genre mereka (cosine similarity). 
#### Memuat Data dalam Format Surprise (Collaborative Based Filtering
Data perlu dimuat dalam format Surprise agar bisa digunakan oleh algoritma collaborative filtering di library Surprise. Format ini memastikan struktur data sesuai standar yang dimengerti oleh sistem rekomendasi berbasis rating. Data perlu dimuat dalam format ini agar dapat diproses oleh algoritma di Surprise seperti SVD atau KNN. Dengan format ini, Surprise dapat mengelola data pengguna dan item secara efisien, melakukan pelatihan, validasi, dan evaluasi model rekomendasi secara optimal.
#### Split data (Collaborative Based Filtering)
Split data adalah pembagian data menjadi dua yaitu train yang digunakan dalam proses pelatihan data sementara test digunakan untuk proses evaluasi. sejumlah 80 persen dari dataset atau sejumlah 8000 data menjadi data train sementara sisanya menjadi data test. 

### Model Development and Result
#### Model Cosine Similarity (Content Based Filtering)
Dalam proyek sistem rekomendasi ini, model dibangun menggunakan cosine similarity. Cosine similarity mengukur sudut antara dua vektor. Semakin kecil sudutnya (semakin mendekati 0 derajat), semakin tinggi nilai cosine similarity (mendekati 1), dan semakin mirip kedua item tersebut. Dalam hal ini film yang memiliki kesamaan genre akan direkomendasikan kepada pengguna. Setelah model dibangun, untuk menguji model dilakukan input dengan judul film yang nantinya akan menghasilkan output berupa rekomendasi filmnya. Berikut adalah top 5 dari rekomendasi filmnya dari film input ('Bloodsuckers': Action|Horror|Sci-Fi): 
- The Story of Robin Hood and His Merrie Men (Genre: Action|Adventure|Children)
- One Flew Over the Cuckoo's Nest (Genre: Drama)
- First Strike (Police Story 4: First Strike) (Ging chaat goo si 4: Ji gaan daan yam mo) (Genre: Action|Adventure|Comedy|Thriller)
- Seuls Two (Genre: Comedy)
- Fade to Black (Genre: Documentary)
Berdasarkan hasilnya, model ini memberikan 2 film rekomendasi yang revelan yaitu film The Story of Robin Hood and His Merrie Men dan First Strike mengingat konten pada film tersebut mengandung genre action. Sementara 3 film lainnya tidak relevan.

#### Model Collaborative Filtering
Dalam collaborative filtering ini, digunakan 4 model sehingga dapat dikomparasikan model mana yang tepat dalam merekomendasikan film berdasarkan pengguna dan rating. Berikut adalah model yang digunakan beserta hasil evaluasinya: 
-	SVD :  metode matrix factorization yang memecah matrix user-item ke dalam representasi laten. SVD mencoba menemukan pola hubungan tersembunyi antara pengguna dan item berdasarkan preferensi (rating). Terdapat 2 model SVD yaitu SVD default dan SVD best tune. Berdasarkan hasil evaluasi, didapatkan bahwa SVD best tune memiliki RMSE dan MAE yang sangat rendah dibandingkan model lainnya sehingga lebih akurat dan memiliki kesalahan prediksi yang kecil. 

-	KNN-User Based dan Item Based : Model User Based mencari pengguna-pengguna yang mirip dengan pengguna target, lalu merekomendasikan item berdasarkan rating dari pengguna-pengguna tetangga (neighbors). Sedangkan Item Based mencari item-item yang mirip (berdasarkan rating pengguna), dan memberikan rekomendasi berdasarkan item serupa yang disukai oleh user. Hasil evaluasi menunjukkan RMSE dan MAE yang sama. Model ini dikatakan kurang mampu memberikan rekomendasi yang akurat dan performa yang masih rendah jika dibandingkan dengan SVD.

Dalam setiap model juga digunakan hyperparameter sebagai berikut:
1.	SVD
-	n_factors : Jumlah latent features atau dimensi faktor tersembunyi user-item. Makin besar nilainya, makin kompleks pola yang bisa dipelajari, tapi risiko overfitting juga lebih tinggi. Nilai yang digunakan adalah [50,100]
-	n_epoch : Jumlah iterasi pelatihan. Semakin besar nilainya, semakin lama model dilatih, yang bisa meningkatkan akurasi sampai titik tertentu. Nilai yang digunakan adalah [20, 30]
-	Lr_all : Learning rate (kecepatan pembelajaran) untuk semua parameter. Nilai yang lebih tinggi mempercepat konvergensi, tapi bisa membuat model tidak stabil. Nilai yang digunakan adalah [0.005, 0.01]
-	Reg_all : Regularisasi untuk semua parameter, berfungsi untuk mencegah overfitting dengan menghukum parameter besar. Semakin besar nilainya, model akan lebih konservatif. Nilai yang digunakan adalah [0.02, 0.1]. 

2.	KNN User/Item 
-	k : Jumlah tetangga terdekat yang akan dipertimbangkan dalam prediksi. Misal: k=40, maka 40 user/item paling mirip digunakan.
-	Min_k : Minimum jumlah tetangga yang diperlukan untuk memberikan prediksi. Jika tidak ada minimal min_k user/item yang bisa dihitung, maka prediksi tidak diberikan. Nilai yang digunakan adalah 1. 
-	Sim_option : Parameter untuk konfigurasi fungsi kemiripan. True untuk user based dan false untuk item based.

Berikut merupakan top 5 rekomendasi menggunakan collaborative based filtering dengan input pengguna (User ID: 99476) dengan best model svd best tune: 
|index|movieId|Predicted Rating|title|userId|
|---|---|---|---|---|
|0|7153|4\.43890378461733|Lord of the Rings: The Return of the King, The|99476|
|1|318|4\.396753927432029|Shawshank Redemption, The|99476|
|2|50|4\.277566656422076|Usual Suspects, The|99476|
|3|608|4\.264900239929637|Fargo|99476|
|4|260|4\.246616285075976|Star Wars: Episode IV - A New Hope|99476|


### Evaluasi
#### Evaluasi Model Pertama (Cosine Similarity-Content Based Filtering) 
Metrik evaluasi yang digunakan dalam proyek ini adalah sebagai berikut: 
1. Precision mengukur seberapa akurat sistem dalam merekomendasikan item yang relevan. Dalam kasus ini, precision 0.12 berarti bahwa dari semua film yang direkomendasikan oleh sistem untuk film Bloodsuckers, sekitar 12% yang benar-benar relevan atau memiliki genre yang sama dengan film input.
2. Recall mengukur seberapa lengkap sistem dalam merekomendasikan semua item yang relevan. Recall 0.13 berarti bahwa sistem hanya merekomendasikan sekitar 13% dari semua film yang relevan atau memiliki genre yang sama dengan film Bloodsuckers.
   
#### Evaluasi Model Kedua (SVD-KNN – Collaborative Based Filtering)
Dalam collaborative filtering ini, digunakan 4 model sehingga dapat dikomparasikan model mana yang tepat dalam merekomendasikan film berdasarkan pengguna dan rating. Berikut adalah model yang digunakan beserta hasil evaluasinya: 
1.	SVD :  metode matrix factorization yang memecah matrix user-item ke dalam representasi laten. SVD mencoba menemukan pola hubungan tersembunyi antara pengguna dan item berdasarkan preferensi (rating). Terdapat 2 model SVD yaitu SVD default dan SVD best tune. Berdasarkan hasil evaluasi, didapatkan bahwa SVD best tune memiliki RMSE dan MAE yang sangat rendah dibandingkan model lainnya sehingga lebih akurat dan memiliki kesalahan prediksi yang kecil. Berikut adalah hasil evaluasinya: 
- [ SVD Default ]
  - RMSE: 1.0567
  - RMSE: 1.0566604089223954
  - MAE:  0.8365
  - MAE: 0.8364668912839488


- [ SVD Best Tuned ]
  - RMSE: 1.0526
  - Best RMSE: 1.0526260689212996
  - MAE:  0.8319
  - Best MAE: 0.8319037980610949

2.	KNN-User Based dan Item Based : Model User Based mencari pengguna-pengguna yang mirip dengan pengguna target, lalu merekomendasikan item berdasarkan rating dari pengguna-pengguna tetangga (neighbors). Sedangkan Item Based mencari item-item yang mirip (berdasarkan rating pengguna), dan memberikan rekomendasi berdasarkan item serupa yang disukai oleh user. Hasil evaluasi menunjukkan RMSE dan MAE yang sama. Model ini dikatakan kurang mampu memberikan rekomendasi yang akurat dan performa yang masih rendah jika dibandingkan dengan SVD. Berikut adalah hasil evaluasinya: 

- [ KNN User-Based ]
  - RMSE: 1.0892
  - RMSE: 1.089230363677147
  - MAE:  0.8706
  - MAE: 0.8706336875

- [ KNN Item-Based ]
  - RMSE: 1.0892
  - RMSE: 1.089230363677147
  - MAE:  0.8706
  - MAE: 0.8706336875

#### Kesimpulan
1. Berdasarkan rumusan masalah pertama, model berhasil dibangun dengan pendekatan content based filtering. Model ini merekomendasikan film berbasis genre yang dipilih pengguna. Apabila pengguna memasukkan film kemudian sistem akan merekomendasikan genre yang mirip dengan film yang diinput pengguna. Namun hasil evaluasi menyatakan bahwa Content based filtering masih rendah dalam hal merekomendasikan film yang sesuai dengan genrenya. Dari tampilan top 5 rekomendasi, hanya dua film yang relevan yang direkomendasikan. Sementara itu hasil evaluasi menggunakan recall dan precision menghasilkan angka yang rendah yang berarti sistem rekomendasi ini masih buruk. Hal ini membuat goals pertama tidak tercapai atau terpenuhi. 
   
2. Mengingat goals pertama (model content based filtering) mempunyai hasil yang buruk, kemudian dilakukan pembangunan model menggunakan pendekatan collaborative filtering dengan algoritma SVD dan KNN (user based dan item based). Hasil menunjukkan bahwa performa model SVD best tune merupakan yang terbaik karena memiliki RMSE dan MAE yang rendah diantara model yang lainnya sehingga layak digunakan untuk dilakukan deployment. 

Referensi:
- Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 13.
- Schwartz, Barry (2004). The Paradox Of Choice. New York, United States: Harper Perennial. ISBN 0-06-000568-8.
- Qin, Jesse. (2024). Netflix’s Billion-Dollar Secret: How Recommendation Systems Fuel Revenue and Innovation. https://www.linkedin.com/pulse/netflixs-billion-dollar-secret-how-recommendation-systems-qin-phd-7zece/
