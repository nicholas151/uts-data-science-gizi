# Estimasi Kandungan Kalori Makanan Indonesia dari Protein, Lemak, dan Karbohidrat Menggunakan Regresi Linear dengan Pembanding SVR

## Gambaran Proyek
Proyek ini memperbaiki dan melengkapi repo `uts-data-science-gizi` menjadi proyek akhir Data Science (Progress 1–4) menggunakan alur **CRISP-DM**. Targetnya adalah memprediksi `calories` (kkal; satuan mengikuti dataset, **belum terverifikasi**) dari fitur makronutrien `proteins`, `fat`, dan `carbohydrate` pada dataset makanan Indonesia `nutrition.csv` (1.346 baris × 7 kolom mentah).

Model utama adalah **Linear Regression (OLS)** (`sklearn.linear_model.LinearRegression`, dengan inferensi/diagnostik `statsmodels.OLS`), dan pembandingnya adalah **SVR** (`sklearn.svm.SVR`) — SVM untuk regresi, **bukan SVC**.

> **Disclaimer:** Estimasi model ini bukan pengganti data gizi resmi, label pangan resmi, atau saran tenaga kesehatan/diet klinis.

## Tujuan SMART (ditetapkan sebelum melihat data test)
| Tujuan | Spesifik | Terukur | Dapat dicapai | Relevan | Batas waktu | Status |
|---|---|---|---|---|---|---|
| T1 Akurasi LR | Prediksi kalori makanan Indonesia | R² test ≥ 0,80; MAE test ≤ 45 kkal; RMSE test ≤ 70 kkal | 3 fitur makro + pipeline reproducible | Estimasi kalori cepat | Sesuai jadwal Progress 4 | ✅ |
| T2 Perbandingan | LR vs SVR | Metrik sama pada split sama | SVR dengan scaling + GridSearchCV pada train | Dasar pemilihan model | Sesuai jadwal Progress 4 | ✅ |
| T3 Kualitas/interpretabilitas | Hasil dapat dijelaskan | Koefisien OLS, diagnostik asumsi, analisis error ≥10 kasus | statsmodels + audit data | Keputusan stakeholder | Sesuai jadwal Progress 4 | ✅ |

## Sumber Data dan Lisensi
- **File:** `data/raw/nutrition.csv` (salinan mentah, tidak diubah).
- **Sumber repo:** https://github.com/nicholas151/uts-data-science-gizi
- **Lisensi/sumber asli:** **belum terverifikasi**. README repo asal tidak mencantumkan lisensi atau sumber asli dataset.
- **Kolom `image`:** URL gambar pihak ketiga; tidak dipakai untuk modeling dan tidak diunduh.
- **Cara verifikasi:** telusuri riwayat commit/README repo asal, cari metadata dataset publik, atau hubungi pemilik repo.

## Metodologi CRISP-DM
```text
[1. Business Understanding] -> [2. Data Understanding/EDA] -> [3. Audit, Cleaning, Preprocessing]
                                                                        |
[5. Evaluation & Kesimpulan] <- [4. Modeling: Linear Regression vs SVR] <-
```

| Tahap | Notebook | Output utama |
|---|---|---|
| Business Understanding | `notebooks/01_business_understanding.ipynb` | `outputs/business_understanding_summary.csv` |
| Data Understanding & EDA | `notebooks/02_data_understanding_eda.ipynb` | `outputs/audit_kualitas.csv`, `outputs/statistik_deskriptif.csv`, `outputs/korelasi_pearson.csv`, figur EDA |
| Audit/Cleaning/Preprocessing | `notebooks/03_audit_cleaning_preprocessing.ipynb` | `data/processed/*`, `outputs/cleaning_log.csv`, `outputs/perbandingan_sebelum_sesudah_cleaning.csv`, `outputs/feature_selection.csv` |
| Modeling | `notebooks/04_modeling.ipynb` | `outputs/ablation_linear_regression.csv`, `outputs/svr_gridsearch_results.csv`, `outputs/perbandingan_model_awal.csv`, `outputs/ols_coefficients.csv` |
| Evaluation | `notebooks/05_evaluation.ipynb` | `outputs/metrik_akhir_test.csv`, `outputs/train_vs_test.csv`, `outputs/status_target_smart.csv`, `outputs/contoh_error.csv`, figur evaluasi |

## Hasil Utama (dari `outputs/`)

### Perbandingan Model pada Test Set
| Model | R² | Adj. R² | MAE (kkal) | RMSE (kkal) | MAPE nonzero (%) |
|---|---:|---:|---:|---:|---:|
| **Linear Regression** | **0.9755** | **0.9752** | **11.50** | **24.64** | **15.09** |
| **SVR** | **0.9731** | **0.9728** | **10.98** | **25.82** | **13.50** |

### Train vs Test (Overfitting)
| Model | R² Train | R² Test | RMSE Train | RMSE Test | ΔR² | Rasio RMSE test/train |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.9461 | 0.9755 | 37.26 | 24.64 | -0.0294 | 0.6612 |
| SVR | 0.9514 | 0.9731 | 35.38 | 25.82 | -0.0217 | 0.7296 |

**Model terbaik:** **Linear Regression**. Berdasarkan RMSE test, Linear Regression (24.64 kkal) sedikit lebih baik daripada SVR (25.82 kkal), dengan R² test sangat tinggi (0.9755). SVR memiliki MAE sedikit lebih rendah (10.98 vs 11.50), tetapi kurang interpretabel dan memerlukan tuning/scaling. Karena selisih performa kecil dan Linear Regression lebih transparan (koefisien dekat faktor Atwater 4/9/4), Linear Regression dipilih sebagai model utama.

**Cek stabilitas (Cek 1):** 5-fold CV pada seluruh data bersih (`KFold(5, shuffle=True, random_state=42)`) menghasilkan LR R² 0,9489±0,0301, MAE 12,72±1,81, RMSE 34,27±8,70; SVR R² 0,9485±0,0296, MAE 11,48±1,85, RMSE 34,49±8,66. CV mean lebih dekat ke train (R² 0,9461) daripada test (R² 0,9755), sehingga R² test > train kemungkinan akibat **variasi split** (test lebih mudah: outlier IQR kalori test 2 vs train 10; residual besar banyak di train). Lihat `outputs/cv_5fold.csv`.

**Sensitivity cleaning (Cek 2):** Baseline LR pada data mentah sebelum cleaning: R² 0,8974, MAE 40,03, RMSE 52,84; setelah cleaning: R² 0,9755, MAE 11,50, RMSE 24,64 (`outputs/perbandingan_sebelum_sesudah_cleaning.csv`). Evaluasi model final pada test mentah (test bersih + 11 baris dibuang): LR turun drastis (R² 0,0414, MAE 28,18, RMSE 168,37), SVR lebih tahan (R² 0,8984, MAE 17,93, RMSE 54,83) (`outputs/sensitivity_test_mentah.csv`). Metrik data bersih bisa lebih optimistis daripada data lapangan kotor.

## Status Target SMART
Catatan transparansi (Cek 3, jujur): target lolos dengan margin besar karena `calories` secara gizi hampir merupakan fungsi linear dari makronutrien (faktor Atwater 4/9/4 kkal/g), sehingga tugas ini relatif mudah untuk model linear. Target **tidak diubah** setelah melihat hasil.
| Target | Nilai aktual LR | Nilai aktual SVR | Status LR | Status SVR |
|---|---:|---:|:---:|:---:|
| R² test ≥ 0,80 | 0.9755 | 0.9731 | ✅ | ✅ |
| MAE test ≤ 45 kkal | 11.50 | 10.98 | ✅ | ✅ |
| RMSE test ≤ 70 kkal | 24.64 | 25.82 | ✅ | ✅ |

## Struktur Repo
```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/nutrition.csv
│   └── processed/
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding_eda.ipynb
│   ├── 03_audit_cleaning_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── archive/
├── outputs/
└── reports/figures/
```

## Cara Menjalankan
Lingkungan diuji dengan Python 3.14 (Windows) dan paket yang terpasang di environment kerja.

```bash
# 1) Buat environment
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

# 2) Install dependensi
pip install -r requirements.txt

# 3) Jalankan notebook berurutan dari root repo
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_business_understanding.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_understanding_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_audit_cleaning_preprocessing.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_modeling.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluation.ipynb
```

Catatan: jalankan dari root repo karena notebook memakai path relatif `Path.cwd().resolve().parents[0]`.

## Matriks Pemenuhan Capaian
| Progress | Capaian | Notebook | Bagian | Status |
|---|---|---|---|---|
| 1 | 1. Judul spesifik & ruang lingkup tepat | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 1]` | ✅ |
| 1 | 2. Judul selaras tujuan & metrik | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 2]` | ✅ |
| 1 | 3. Pernyataan masalah & stakeholder | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 3]` | ✅ |
| 1 | 4. Tujuan SMART | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 4]` | ✅ |
| 1 | 5. Metrik keberhasilan | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 5]` | ✅ |
| 1 | 6. Asumsi & batasan | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 6]` | ✅ |
| 1 | 7. Risiko & etika + mitigasi | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 7]` | ✅ |
| 1 | 8. Nilai bisnis/use-case | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 8]` | ✅ |
| 1 | 9. Pemetaan stakeholder–data | `01_business_understanding.ipynb` | `[Progress 1 – Capaian 9]` | ✅ |
| 2 | 1. Sumber data & lisensi/izin | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 1]` | ✅ (lisensi belum terverifikasi) |
| 2 | 2. Data dictionary | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 2]` | ✅ |
| 2 | 3. Kualitas data | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 3]` | ✅ |
| 2 | 4. EDA ber-insight | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 4]` | ✅ |
| 2 | 5. Representativitas & bias sampel | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 5]` | ✅ |
| 2 | 6. Statistik deskriptif | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 6]` | ✅ |
| 2 | 7. Visualisasi fitur & target | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 7]` | ✅ |
| 2 | 8. Korelasi fitur–target | `02_data_understanding_eda.ipynb` | `[Progress 2 – Capaian 8]` | ✅ |
| 2 | 9. Audit kualitas data (asumsi) | `03_audit_cleaning_preprocessing.ipynb` | `[Progress 2 – Capaian 9]` | ✅ |
| 2 | 10. Data cleaning (asumsi) | `03_audit_cleaning_preprocessing.ipynb` | `[Progress 2 – Capaian 10]` | ✅ |
| 2 | 11. Perbandingan sebelum/sesudah (asumsi) | `03_audit_cleaning_preprocessing.ipynb` | `[Progress 2 – Capaian 11]` | ✅ |
| 2 | 12. Preprocessing (asumsi) | `03_audit_cleaning_preprocessing.ipynb` | `[Progress 2 – Capaian 12]` | ✅ |
| 2 | 13. Split train/test (asumsi) | `03_audit_cleaning_preprocessing.ipynb` | `[Progress 2 – Capaian 13]` | ✅ |
| 2 | 14. Seleksi fitur (asumsi) | `03_audit_cleaning_preprocessing.ipynb` | `[Progress 2 – Capaian 14]` | ✅ |
| 3 | 1. Pemilihan model | `04_modeling.ipynb` | `[Progress 3 – Capaian 1]` | ✅ |
| 3 | 2. Alasan tiap model | `04_modeling.ipynb` | `[Progress 3 – Capaian 2]` | ✅ |
| 3 | 3. Proses training | `04_modeling.ipynb` | `[Progress 3 – Capaian 3]` | ✅ |
| 3 | 4. Reproducibility | `04_modeling.ipynb` | `[Progress 3 – Capaian 4]` | ✅ |
| 3 | 5. Pengaturan parameter | `04_modeling.ipynb` | `[Progress 3 – Capaian 5]` | ✅ |
| 3 | 6. Hasil awal model | `04_modeling.ipynb` | `[Progress 3 – Capaian 6]` | ✅ |
| 3 | 7. Tabel perbandingan | `04_modeling.ipynb` | `[Progress 3 – Capaian 7]` | ✅ |
| 3 | 8. Arti hasil awal | `04_modeling.ipynb` | `[Progress 3 – Capaian 8]` | ✅ |
| 4 | 1. Metrik akhir di data test | `05_evaluation.ipynb` | `[Progress 4 – Capaian 1]` | ✅ |
| 4 | 2. Grafik pendukung | `05_evaluation.ipynb` | `[Progress 4 – Capaian 2]` | ✅ |
| 4 | 3. Train vs test (overfitting) | `05_evaluation.ipynb` | `[Progress 4 – Capaian 3]` | ✅ |
| 4 | 3a. Cek 1: 5-fold CV & diagnosis test>train | `05_evaluation.ipynb` | `Cek 1 (Tambahan)` | ✅ |
| 4 | 9a. Cek 2: Sensitivity test mentah | `05_evaluation.ipynb` | `Cek 2 (Tambahan)` | ✅ |
| 4 | 4. Model terbaik | `05_evaluation.ipynb` | `[Progress 4 – Capaian 4]` | ✅ |
| 4 | 5. Analisis kesalahan | `05_evaluation.ipynb` | `[Progress 4 – Capaian 5]` | ✅ |
| 4 | 6. Jawaban pertanyaan BU | `05_evaluation.ipynb` | `[Progress 4 – Capaian 6]` | ✅ |
| 4 | 7. Metrik bahasa sehari-hari | `05_evaluation.ipynb` | `[Progress 4 – Capaian 7]` | ✅ |
| 4 | 8. Target tercapai/tidak | `05_evaluation.ipynb` | `[Progress 4 – Capaian 8]` | ✅ |
| 4 | 9. Keterbatasan | `05_evaluation.ipynb` | `[Progress 4 – Capaian 9]` | ✅ |

## Catatan Keputusan
1. **Capaian Progress 2 nomor 9–14 diasumsikan** karena daftar pengguna terpotong pada "Audit...". Saya mengikuti alur audit, cleaning, perbandingan sebelum/sesudah, preprocessing, split, seleksi fitur (pola notebook lama) dan menyesuaikannya ke regresi.
2. **`kategori_kalori` tidak dipakai** karena proyek diubah dari klasifikasi ke regresi; kolom ini turunan target.
3. **`energi_atwater` tidak jadi fitur model utama** karena hampir deterministik terhadap kalori; dipakai untuk audit, baseline pembanding, dan analisis error.
4. **Split acak stratified 80/20, `random_state=42`** dipakai karena data cross-sectional (bukan time series).
5. **11 baris `macro_sum > 100 g` dihapus** karena tidak fisik untuk komposisi per 100 g/satuan sebanding; tercatat di `outputs/cleaning_log.csv`.
6. **`Agar-agar` (`calories==0`) dipertahankan** sebagai kasus valid langka; MAPE tidak dijadikan metrik utama.
7. **Nama duplikat ternormalisasi dipertahankan** (ditandai) karena tanpa sumber eksternal tidak bisa dipastikan baris mana yang benar.
8. **Lisensi dataset belum terverifikasi**; README/LICENSE tidak mengklaim lisensi data.
9. **Modul/minggu praktikum asal model** belum dikonfirmasi pengguna; ditulis sebagai placeholder di notebook 04.
10. **Tanggal tenggat Progress 4** belum dikonfirmasi; target waktu ditulis "sesuai jadwal Progress 4 mata kuliah".
11. **Cek 1:** CV 5-fold seluruh data bersih memberi LR R² 0,9489±0,0301, MAE 12,72±1,81, RMSE 34,27±8,70; SVR R² 0,9485±0,0296, MAE 11,48±1,85, RMSE 34,49±8,66. R² test > train dijelaskan oleh variasi split/outlier train.
12. **Cek 2:** 11 baris dibuang saat cleaning (`macro_sum>100g`) tidak masuk test bersih; sensitivity (test bersih+baris dibuang) membuat LR turun drastis (R² 0,0414, RMSE 168,37) dan SVR turun (R² 0,8984, RMSE 54,83), sehingga metrik data bersih lebih optimistis.
13. **Cek 3:** Target SMART lolos margin besar karena hubungan Atwater hampir linear; target tidak diubah.
