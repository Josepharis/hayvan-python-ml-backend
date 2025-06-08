# 🐄 Hayvancılık Gelişim Analizi - Python Backend

Bu Python backend, scikit-learn kullanarak hayvan gelişim tahminleri yapar.

## 📋 Kurulum

1. **Python ortamı hazırlayın:**
```bash
cd python_backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Firebase Service Account ayarlayın:**
   - Firebase Console'dan service account key indirin
   - `firebase-service-account.json` olarak kaydedin

4. **Sunucuyu başlatın:**
```bash
python main.py
```

## 🤖 Makine Öğrenmesi Modeli

### Kullanılan Algoritmalar:
- **Random Forest Regressor**: Kilo tahmini için
- **Random Forest Regressor**: Sağlık skoru tahmini için  
- **StandardScaler**: Veri normalizasyonu için

### Özellikler (Features):
- Hayvan yaşı (gün)
- Cinsiyet (0/1)
- Tür (one-hot encoding)
- Son kilo
- Ortalama kilo
- Kilo trendi
- Son boy
- Ortalama boy
- Kayıt sayısı

### Tahmin Çıktıları:
- **3, 6, 12, 24 aylık kilo tahminleri**
- **Sağlık skoru (0-100)**
- **Gelişim trendi** (pozitif/negatif/kararsız)
- **Öneriler listesi**
- **Risk faktörleri**

## 🔄 API Endpoints

### `POST /predict/{hayvan_id}`
Belirli bir hayvan için gelişim tahmini yapar.

**Response:**
```json
{
  "hayvanId": "abc123",
  "gelisimTrendi": "pozitif",
  "tahminKilo": {
    "3ay": 45.2,
    "6ay": 65.8,
    "12ay": 98.5,
    "24ay": 145.3
  },
  "saglikSkoru": 85.6,
  "onerimler": [
    "Günlük besleme miktarını 2kg artırın",
    "Haftada 2 kez veteriner kontrolü yapın"
  ],
  "riskFaktorleri": [
    "Mevsimsel hastalık riski orta seviyede"
  ]
}
```

### `POST /retrain`
Modelleri Firebase'deki güncel verilerle yeniden eğitir.

### `GET /health`
API sağlık durumunu kontrol eder.

## 📊 Model Eğitimi

Model otomatik olarak:
1. Firebase'den tüm hayvan verilerini çeker
2. En az 2 gelişim kaydı olan hayvanları kullanır
3. Feature extraction yapar
4. Random Forest modellerini eğitir
5. Modelleri `*.pkl` dosyalarına kaydeder

## 🎯 Gelişim Tahmin Algoritması

1. **Veri Toplama**: Firebase'den hayvan + gelişim kayıtları
2. **Feature Engineering**: Yaş, tür, cinsiyet, kilo trendleri
3. **Model Prediction**: Random Forest ile tahmin
4. **Rule-based Fallback**: ML modeli yoksa kural tabanlı
5. **Öneri Sistemi**: Sağlık skoru ve risk analizi

## 🚀 Production Deployment

1. **Docker ile:**
```bash
docker build -t hayvan-ml .
docker run -p 8000:8000 hayvan-ml
```

2. **Google Cloud Run / Heroku / AWS ile deploy**

3. **Firebase Cloud Functions ile entegrasyon** 