# Sistem Rekomendasi Film dengan Menggunakan Content Based Filtering dan Collaborative Based Filtering

## Project Overview
Industri kreatif film menunjukkan pertumbuhan signifikan seiring dengan meningkatnya konsumsi media digital di era modern. Digitalisasi turut mendorong transformasi besar dalam distribusi film melalui platform streaming berbasis website seperti Netflix, Disney+, dan lainnya. Kemudahan akses ini menciptakan tantangan baru berupa information overload, di mana pengguna kesulitan memilih film yang sesuai dengan preferensi mereka.
Menurut laporan Netflix, sistem rekomendasi mereka berkontribusi langsung terhadap penghematan lebih dari $1 miliar per tahun dengan mengurangi tingkat churn pelanggan, serta menjaga tingkat retensi pengguna pada angka mengesankan sekitar 2,3% (Gomez-Uribe & Hunt, 2015). Oleh karena itu, sistem rekomendasi berbasis content-based filtering menjadi solusi yang relevan untuk mempersonalisasi pengalaman pengguna dan meningkatkan kepuasan mereka dalam menjelajahi katalog film.
Referensi:
Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 13.

### Business Understanding
Sebuah platform distribusi film digital baru yang diluncurkan di Indonesia beberapa bulan terakhir mengalami tantangan serius dalam mempertahankan jumlah pelanggannya. Setelah awal peluncuran yang menjanjikan, data menunjukkan terjadinya penurunan jumlah subscriber secara bertahap. Berdasarkan analisis ulasan dan umpan balik pengguna, salah satu masalah utama yang muncul adalah kebingungan dalam memilih film untuk ditonton. Meski koleksi film yang tersedia di platform sangat beragam dan banyak, justru hal tersebut menciptakan beban kognitif bagi pengguna dalam membuat keputusan.
Fenomena ini dikenal sebagai paradox of choice—di mana semakin banyak pilihan yang tersedia, semakin besar pula kebingungan yang dialami pengguna. Konsep ini diperkenalkan oleh psikolog Barry Schwartz dalam bukunya "The Paradox of Choice: Why More Is Less", yang menjelaskan bahwa terlalu banyak pilihan dapat menyebabkan kecemasan, penundaan keputusan, bahkan penyesalan setelah memilih. Dalam konteks platform film digital, hal ini berpotensi menurunkan kepuasan pengguna dan meningkatkan risiko churn (berhentinya pelanggan).
Masalah ini semakin diperparah dengan tidaknya tersedia sistem rekomendasi yang personal. Saat ini, film yang ditampilkan di halaman utama cenderung didasarkan pada popularitas umum, bukan preferensi individual pengguna. Akibatnya, pengalaman menonton menjadi kurang relevan dan kurang menarik, serta tidak mendorong eksplorasi konten yang lebih luas. Sebagai pembanding, platform seperti Netflix menyatakan bahwa sistem rekomendasi yang baik mampu menghemat lebih dari $1 miliar setiap tahun dengan cara mengurangi tingkat churn melalui peningkatan pengalaman pengguna (Qin, 2024).

Referensi : 
Schwartz, Barry (2004). The Paradox Of Choice. New York, United States: Harper Perennial. ISBN 0-06-000568-8.
Qin, Jesse. (2024, March 15). Netflix's Billion Dollar Secret: How Recommendation Systems Work. LinkedIn Pulse. https://www.linkedin.com/pulse/netflixs-billion-dollar-secret-how-recommendation-systems-qin-phd-7zece/

### Problem Statement
Dari studi kasus ini, pernyataan masalah dapat dirumuskan sebagai berikut: 
1.	Pengguna kesulitan menemukan film yang sesuai dengan preferensinya akibat banyaknya pilihan film yang tersedia di platform streaming. 
2.	Sistem rekomendasi yang telah dibangun ternyata tidak sepenuhnya tepat merekomendasikan film yang sesuai dengan preferensi pengguna. 

### Goals
1.	Terbentuk sistem rekomendasi berbasis content based filtering yang dapat merekomendasikan film yang sesuai dengan minat genre film pengguna. 
2.	Meningkatkan akurasi dan relevansi sistem rekomendasi film dengan menerapkan pendekatan atau metode yang lebih efektif dalam memahami preferensi pengguna.

### Solution Statement:
1.	Untuk mengatasi masalah 1, maka dibangun sistem rekomendasi film dengan pendekatan content based filtering. Sistem ini menggunakan TF-IDF Vectorizer pada fitur genres untuk menghitung kemiripan antar film dengan Cosine Similarity (Shrivastava dan Sisodia, 2019).
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
1.	Membersihkan kolom tittle dari double spasi, menghilangkan teks selain huruf. 
2.	Menghapus kolom yang tidak relevan, dalam hal ini kolom yang dimaksud adalah kolom index. 
3.	Menghapus tanda pemisah antar genre pada kolom genre sehingga dapat mempermudah dalam melakukan visualisasi data.
4.	Menggunakan data sampling dengan n sebanyak 10.000 baris data. Sampling digunakan mengingat keterbatasan RAM. Penggunaan matrik yang lebih besar berpotensi menyebabkan terjadinya crush pada runtimenya. 
5.	Melakukan vektorisasi dengan TF-IDF pada kolom genre yang bertujuan untuk mengubah teks menjadi representasi numerik (vektor) yang dapat dipahami oleh model machine learning.

## Model Development
1.	Model Cosine Similarity (Content Based Filtering)
Dalam proyek sistem rekomendasi ini, model dibangun menggunakan cosine similarity. Cosine similarity mengukur sudut antara dua vektor. Semakin kecil sudutnya (semakin mendekati 0 derajat), semakin tinggi nilai cosine similarity (mendekati 1), dan semakin mirip kedua item tersebut. Dalam hal ini film yang memiliki kesamaan genre akan direkomendasikan kepada pengguna. Setelah model dibangun, untuk menguji model dilakukan input dengan judul film yang nantinya akan menghasilkan output berupa rekomendasi filmnya. 

2.	Model Collaborative Filtering
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

## Evaluasi
### Evaluasi Model Pertama (Cosine Similarity-Content Based Filtering) 
Metrik evaluasi yang digunakan dalam proyek ini adalah sebagai berikut: 
1.	Precision mengukur seberapa akurat rekomendasi yang diberikan. Lebih spesifik, Precision menjawab pertanyaan: Dari semua film yang direkomendasikan, berapa proporsi film yang benar-benar relevan?. Precision tinggi menunjukkan bahwa sebagian besar film yang direkomendasikan oleh model memang relevan bagi pengguna. Hasil menunjukkan bahwa precision masih rendah yaitu 0.3, artinya hanya 30 persen sistem merekomendasikan film sesuai dengan preferensi yang dipilih pengguna. 
2.	Recall mengukur seberapa lengkap rekomendasi yang diberikan. Recall menjawab pertanyaan: Dari semua film yang relevan, berapa proporsi film yang berhasil direkomendasikan oleh model?. Recall tinggi menunjukkan bahwa model berhasil merekomendasikan sebagian besar film yang relevan bagi pengguna. Hasil menunjukkan bahwa recall juga rendah yaitu hanya 0.3, artinya hanya 30 persen dari total film yang relevan yang direkomendasikan model.
   
### Evaluasi Model Kedua (SVD-KNN – Collaborative Based Filtering)
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
