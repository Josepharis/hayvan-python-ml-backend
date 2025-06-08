from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import joblib
import os
from typing import Dict, List, Optional
from pydantic import BaseModel
from load_dataset import load_hayvan_dataset, load_gelisim_dataset, prepare_ml_data

app = FastAPI(title="Hayvancılık ML API", version="1.0.0")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
weight_model = None
health_model = None
ml_data = None
feature_columns = None
model_accuracy = {}

class PredictionResponse(BaseModel):
    hayvan_id: str
    predictions: Dict
    health_score: float
    recommendations: List[str]
    risk_factors: List[str]

class ModelInfo(BaseModel):
    model_type: str
    accuracy: float
    data_points: int
    last_trained: str
    features_used: int

@app.on_event("startup")
async def startup_event():
    """
    Uygulama başlatıldığında modelleri yükle ve eğit
    """
    print("🚀 Hayvancılık ML API başlatılıyor...")
    await retrain_models()

@app.get("/")
async def root():
    return {
        "message": "Hayvancılık ML API",
        "version": "1.0.0",
        "dataset_source": "Gerçek araştırma verileri",
        "endpoints": ["/predict/{hayvan_id}", "/retrain", "/health", "/model-info"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": weight_model is not None and health_model is not None,
        "data_points": len(ml_data) if ml_data is not None else 0
    }

@app.get("/model-info", response_model=List[ModelInfo])
async def get_model_info():
    """
    Model bilgilerini döndür
    """
    if weight_model is None:
        raise HTTPException(status_code=500, detail="Modeller henüz eğitilmedi")
    
    return [
        ModelInfo(
            model_type="Weight Prediction",
            accuracy=model_accuracy.get("weight", 0.0),
            data_points=len(ml_data) if ml_data is not None else 0,
            last_trained=datetime.now().isoformat(),
            features_used=len(feature_columns) if feature_columns else 0
        ),
        ModelInfo(
            model_type="Health Score",
            accuracy=model_accuracy.get("health", 0.0),
            data_points=len(ml_data) if ml_data is not None else 0,
            last_trained=datetime.now().isoformat(),
            features_used=len(feature_columns) if feature_columns else 0
        )
    ]

@app.post("/retrain")
async def retrain_models():
    """
    Modelleri gerçek veri seti ile yeniden eğit
    """
    global weight_model, health_model, ml_data, feature_columns, model_accuracy
    
    try:
        print("📊 Veri seti yükleniyor...")
        ml_data, feature_columns, target_columns = prepare_ml_data()
        
        if ml_data is None or len(ml_data) < 10:
            raise HTTPException(status_code=400, detail="Yetersiz veri")
        
        print(f"✅ {len(ml_data)} kayıt yüklendi, {len(feature_columns)} feature kullanılacak")
        
        # Features ve targets hazırla
        X = ml_data[feature_columns]
        y_weight = ml_data['kilo']
        y_health = ml_data['saglik_encoded']
        
        # Train-test split
        X_train, X_test, y_weight_train, y_weight_test = train_test_split(
            X, y_weight, test_size=0.2, random_state=42
        )
        
        _, _, y_health_train, y_health_test = train_test_split(
            X, y_health, test_size=0.2, random_state=42
        )
        
        # Kilo tahmin modeli
        print("🤖 Kilo tahmin modeli eğitiliyor...")
        weight_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        weight_model.fit(X_train, y_weight_train)
        
        # Model performansı
        y_weight_pred = weight_model.predict(X_test)
        weight_r2 = r2_score(y_weight_test, y_weight_pred)
        weight_mae = mean_absolute_error(y_weight_test, y_weight_pred)
        model_accuracy["weight"] = weight_r2
        
        print(f"   R² Score: {weight_r2:.3f}")
        print(f"   MAE: {weight_mae:.2f} kg")
        
        # Sağlık skoru modeli
        print("🏥 Sağlık skoru modeli eğitiliyor...")
        health_model = RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            min_samples_split=3,
            random_state=42
        )
        health_model.fit(X_train, y_health_train)
        
        # Sağlık modeli performansı
        y_health_pred = health_model.predict(X_test)
        health_r2 = r2_score(y_health_test, y_health_pred)
        health_mae = mean_absolute_error(y_health_test, y_health_pred)
        model_accuracy["health"] = health_r2
        
        print(f"   R² Score: {health_r2:.3f}")
        print(f"   MAE: {health_mae:.2f}")
        
        # Modelleri kaydet
        joblib.dump(weight_model, 'weight_model.pkl')
        joblib.dump(health_model, 'health_model.pkl')
        
        print("✅ Modeller başarıyla eğitildi ve kaydedildi!")
        
        return {
            "message": "Modeller başarıyla eğitildi",
            "data_points": len(ml_data),
            "features": len(feature_columns),
            "weight_model": {
                "r2_score": weight_r2,
                "mae": weight_mae
            },
            "health_model": {
                "r2_score": health_r2,
                "mae": health_mae
            }
        }
        
    except Exception as e:
        print(f"❌ Model eğitimi hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model eğitimi hatası: {str(e)}")

@app.get("/predict/{hayvan_id}", response_model=PredictionResponse)
async def predict_growth(hayvan_id: str):
    """
    Belirli bir hayvan için gelişim tahmini yap
    """
    if weight_model is None or health_model is None:
        raise HTTPException(status_code=500, detail="Modeller henüz eğitilmedi")
    
    if ml_data is None:
        raise HTTPException(status_code=500, detail="Veri seti yüklenmedi")
    
    try:
        # Hayvan verilerini bul
        hayvan_data = ml_data[ml_data['hayvanId'] == hayvan_id]
        
        if hayvan_data.empty:
            raise HTTPException(status_code=404, detail="Hayvan bulunamadı")
        
        # En son kayıt
        latest_record = hayvan_data.iloc[-1]
        current_age_months = latest_record['yasAy']
        current_weight = latest_record['kilo']
        
        print(f"🔮 {hayvan_id} için tahmin yapılıyor (Mevcut: {current_age_months} ay, {current_weight:.1f} kg)")
        
        # Gelecek tahminleri için feature'ları hazırla
        base_features = latest_record[feature_columns].values.reshape(1, -1)
        
        predictions = {}
        target_months = [3, 6, 12, 24]
        
        for target_month in target_months:
            if target_month > current_age_months:
                # Yaş günü güncelle (30 gün = 1 ay)
                future_features = base_features.copy()
                age_index = feature_columns.index('yas_gun')
                future_features[0, age_index] = target_month * 30
                
                # Tahmin yap
                predicted_weight = weight_model.predict(future_features)[0]
                
                # Gerçekçi sınırlar uygula
                predicted_weight = max(predicted_weight, current_weight)
                
                predictions[f"{target_month}_month"] = {
                    "age_months": target_month,
                    "predicted_weight": round(predicted_weight, 1),
                    "weight_gain": round(predicted_weight - current_weight, 1),
                    "daily_gain": round((predicted_weight - current_weight) / ((target_month - current_age_months) * 30), 3)
                }
        
        # Sağlık skoru hesapla
        health_score_raw = health_model.predict(base_features)[0]
        health_score = min(100, max(0, (health_score_raw / 4) * 100))  # 0-4 skalasını 0-100'e çevir
        
        # Öneriler ve risk faktörleri
        recommendations = generate_recommendations(latest_record, predictions, health_score)
        risk_factors = identify_risk_factors(latest_record, predictions, health_score)
        
        return PredictionResponse(
            hayvan_id=hayvan_id,
            predictions=predictions,
            health_score=round(health_score, 1),
            recommendations=recommendations,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        print(f"❌ Tahmin hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

def generate_recommendations(record, predictions, health_score):
    """
    Hayvan için öneriler oluştur
    """
    recommendations = []
    
    # Sağlık durumuna göre öneriler
    if health_score < 50:
        recommendations.append("Veteriner kontrolü acil olarak gerekli")
        recommendations.append("Beslenme programını gözden geçirin")
    elif health_score < 70:
        recommendations.append("Sağlık durumunu yakından takip edin")
        recommendations.append("Vitamin desteği değerlendirin")
    else:
        recommendations.append("Sağlık durumu iyi, mevcut bakımı sürdürün")
    
    # Yaşa göre öneriler
    current_age = record['yasAy']
    if current_age < 6:
        recommendations.append("Dana döneminde yüksek kaliteli süt ve mama vermeye devam edin")
    elif current_age < 12:
        recommendations.append("Büyüme döneminde protein açısından zengin yem vermeye özen gösterin")
    else:
        recommendations.append("Yetişkin beslenme programına geçiş yapabilirsiniz")
    
    # Mevsimsel öneriler
    current_season = record.get('mevsim_encoded', 1)
    if current_season == 0:  # Kış
        recommendations.append("Kış aylarında ek kalori desteği sağlayın")
    elif current_season == 2:  # Yaz
        recommendations.append("Sıcak havalarda gölge ve temiz su sağlayın")
    
    return recommendations

def identify_risk_factors(record, predictions, health_score):
    """
    Risk faktörlerini belirle
    """
    risk_factors = []
    
    # Sağlık riski
    if health_score < 60:
        risk_factors.append("Düşük sağlık skoru")
    
    # Yaş riski
    current_age = record['yasAy']
    if current_age > 20:
        risk_factors.append("İleri yaş - daha dikkatli takip gerekli")
    
    # Kilo riski
    current_weight = record['kilo']
    if current_weight < 100 and current_age > 6:
        risk_factors.append("Yaşına göre düşük kilo")
    
    # Gelecek tahmin riski
    if predictions:
        # En yakın tahmindeki günlük artış kontrolü
        for period, pred in predictions.items():
            if pred['daily_gain'] < 1.0:
                risk_factors.append(f"Düşük büyüme hızı tahmini ({period})")
                break
    
    # Çevresel riskler
    temperature = record.get('sicaklik', 25)
    humidity = record.get('nem', 50)
    
    if temperature > 35:
        risk_factors.append("Yüksek sıcaklık stresi")
    elif temperature < 5:
        risk_factors.append("Düşük sıcaklık stresi")
    
    if humidity > 80:
        risk_factors.append("Yüksek nem oranı")
    
    return risk_factors

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 ML Server starting on port {port}")
    print(f"📊 Dataset ready with 8000+ records!")
    uvicorn.run(app, host="0.0.0.0", port=port) 