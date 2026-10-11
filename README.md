# Breast_Cancer_Wisconsin_Diagnosis_using_Logistic_Regression

# Logistic Regression dari Nol (From Scratch) - Breast Cancer Classification

Repositori ini berisi implementasi algoritma **Logistic Regression** yang dibangun dari awal (*from scratch*) menggunakan **Python**, **NumPy**, dan **Pandas** untuk melakukan klasifikasi biner pada dataset medis[cite: 1].

## 📋 Daftar Isi
- [Tentang Proyek](#tentang-proyek)
- [Pre-processing Data](#pre-processing-data)
- [Arsitektur Model & Implementasi](#arsitektur-model--implementasi)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Cara Menjalankan](#cara-menjalankan)

---

## 🔍 Tentang Proyek
Proyek ini mendemonstrasikan pemahaman mendalam mengenai prinsip kerja machine learning tanpa bergantung pada pustaka estimator tingkat tinggi untuk modelnya[cite: 1]. Seluruh fungsi matematika utama—mulai dari inisialisasi parameter, fungsi aktivasi, *forward-backward propagation*, hingga *gradient descent*—diimplementasikan secara manual menggunakan matriks NumPy[cite: 1]. Dataset yang digunakan memiliki total 569 baris data dan 33 kolom fitur[cite: 1].

---

## 🧹 Pre-processing Data
Sebelum dimasukkan ke dalam model, data mentah melalui beberapa tahapan pembersihan dan penyiapan:
* Penghapusan kolom yang tidak relevan atau kosong, yaitu `'id'` dan `'Unnamed: 32'`[cite: 1].
* Melakukan *mapping* pada kolom target `diagnosis`, mengubah nilai string (`'M'` untuk ganas/malignant dan `'B'` untuk jinak/benign) menjadi nilai biner numerik `1` dan `0`[cite: 1].
* Melakukan normalisasi fitur (*min-max scaling*) agar seluruh nilai fitur berada pada rentang skala antara 0 dan 1[cite: 1].
* Membagi dataset menjadi data latih dan data uji menggunakan fungsi `train_test_split` dengan rasio `test_size = 0.15` dan `random_state = 42`[cite: 1]. Proses ini menghasilkan 483 sampel data latih dan 86 sampel data uji[cite: 1].

---

## ⚙️ Arsitektur Model & Implementasi
Model ini terdiri dari beberapa fungsi utama:
* **`initialize_weights_and_bias`**: Menginisialisasi matriks bobot ($w$) dengan nilai acak berukuran kecil (dikalikan 0.01) dan nilai bias ($b$) sebesar 0.0[cite: 1].
* **`sigmoid`**: Fungsi aktivasi logistik untuk memetakan hasil komputasi linear menjadi nilai probabilitas antara 0 dan 1[cite: 1].
* **`forward_backward_propagation`**: Menghitung nilai *cost* (kerugian) menggunakan metode *cross-entropy* serta menghitung gradien turunan terhadap bobot dan bias[cite: 1].
* **`update`**: Memperbarui parameter bobot dan bias secara iteratif menggunakan algoritma *gradient descent* dengan *learning rate* 0.01 selama 1000 iterasi[cite: 1].
* **`predict`**: Menghasilkan label prediksi (0 atau 1) berdasarkan ambang batas (*threshold*) probabilitas sebesar 0.5[cite: 1].

---

## 📈 Hasil Evaluasi
Setelah proses pelatihan berjalan selama 1000 iterasi, model memberikan hasil performa sebagai berikut:
* **Akurasi Data Latih (Train Accuracy):** ~90.68%[cite: 1]
* **Akurasi Data Uji (Test Accuracy):** ~88.37%[cite: 1]

---

## 🚀 Cara Menjalankan
1. Pastikan Anda telah memasang pustaka Python yang diperlukan (`numpy`, `pandas`, `matplotlib`, dan `scikit-learn`)[cite: 1].
2. Letakkan file dataset dengan nama `data.csv` pada direktori yang sama dengan skrip program[cite: 1].
3. Jalankan kode program melalui Jupyter Notebook, Google Colab, atau terminal Python Anda[cite: 1].
