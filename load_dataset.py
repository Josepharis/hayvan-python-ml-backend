import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def load_hayvan_dataset():
    """
    Oluşturulan hayvan veri setini yükler
    """
    csv_path = 'hayvanlar_dataset.csv'
    json_path = 'hayvanlar_dataset.json'
    
    if os.path.exists(csv_path):
        print(f"📊 Hayvan veri seti yükleniyor: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ {len(df)} hayvan kaydı yüklendi")
        return df
    elif os.path.exists(json_path):
        print(f"📊 Hayvan veri seti yükleniyor: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"✅ {len(df)} hayvan kaydı yüklendi")
        return df
    else:
        print("❌ Veri seti bulunamadı! Önce create_sample_dataset.py çalıştırın.")
        return None

def load_gelisim_dataset():
    """
    Oluşturulan gelişim kayıtları veri setini yükler
    """
    csv_path = 'gelisim_kayitlari_dataset.csv'
    json_path = 'gelisim_kayitlari_dataset.json'
    
    if os.path.exists(csv_path):
        print(f"📊 Gelişim veri seti yükleniyor: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✅ {len(df)} gelişim kaydı yüklendi")
        return df
    elif os.path.exists(json_path):
        print(f"📊 Gelişim veri seti yükleniyor: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"✅ {len(df)} gelişim kaydı yüklendi")
        return df
    else:
        print("❌ Veri seti bulunamadı! Önce create_sample_dataset.py çalıştırın.")
        return None

def analyze_dataset():
    """
    Veri setinin detaylı analizini yapar
    """
    print("🔍 VERİ SETİ ANALİZİ")
    print("=" * 50)
    
    # Hayvan verilerini yükle
    df_hayvanlar = load_hayvan_dataset()
    df_gelisim = load_gelisim_dataset()
    
    if df_hayvanlar is None or df_gelisim is None:
        return
    
    # Temel istatistikler
    print(f"\n📊 TEMEL İSTATİSTİKLER:")
    print(f"   Toplam hayvan sayısı: {len(df_hayvanlar)}")
    print(f"   Toplam gelişim kaydı: {len(df_gelisim)}")
    print(f"   Ortalama kayıt/hayvan: {len(df_gelisim)/len(df_hayvanlar):.1f}")
    
    # Irk dağılımı
    print(f"\n🐮 IRK DAĞILIMI:")
    irk_dagilimi = df_hayvanlar['irk'].value_counts()
    for irk, sayi in irk_dagilimi.items():
        yuzde = (sayi / len(df_hayvanlar)) * 100
        print(f"   {irk}: {sayi} baş ({yuzde:.1f}%)")
    
    # Cinsiyet dağılımı
    print(f"\n♂♀ CİNSİYET DAĞILIMI:")
    cinsiyet_dagilimi = df_hayvanlar['cinsiyet'].value_counts()
    for cinsiyet, sayi in cinsiyet_dagilimi.items():
        yuzde = (sayi / len(df_hayvanlar)) * 100
        print(f"   {cinsiyet}: {sayi} baş ({yuzde:.1f}%)")
    
    # Yaş dağılımı
    print(f"\n📈 YAŞ DAĞILIMI (ay):")
    yas_stats = df_gelisim['yasAy'].describe()
    print(f"   Min yaş: {yas_stats['min']:.0f} ay")
    print(f"   Max yaş: {yas_stats['max']:.0f} ay") 
    print(f"   Ortalama yaş: {yas_stats['mean']:.1f} ay")
    print(f"   Medyan yaş: {yas_stats['50%']:.1f} ay")
    
    # Kilo analizi
    print(f"\n⚖️ KİLO ANALİZİ:")
    kilo_stats = df_gelisim['kilo'].describe()
    print(f"   Min kilo: {kilo_stats['min']:.1f} kg")
    print(f"   Max kilo: {kilo_stats['max']:.1f} kg")
    print(f"   Ortalama kilo: {kilo_stats['mean']:.1f} kg")
    print(f"   Medyan kilo: {kilo_stats['50%']:.1f} kg")
    
    # Günlük artış analizi
    print(f"\n📊 GÜNLÜK AĞIRLIK ARTIŞI:")
    artis_data = df_gelisim[df_gelisim['gunlukArtis'] > 0]['gunlukArtis']
    if len(artis_data) > 0:
        artis_stats = artis_data.describe()
        print(f"   Min artış: {artis_stats['min']:.3f} kg/gün")
        print(f"   Max artış: {artis_stats['max']:.3f} kg/gün")
        print(f"   Ortalama artış: {artis_stats['mean']:.3f} kg/gün")
        print(f"   Medyan artış: {artis_stats['50%']:.3f} kg/gün")
    
    # Irka göre performans
    print(f"\n🏆 IRKA GÖRE PERFORMANS:")
    for irk in df_hayvanlar['irk'].unique():
        irk_hayvanlar = df_hayvanlar[df_hayvanlar['irk'] == irk]['id'].tolist()
        irk_gelisim = df_gelisim[df_gelisim['hayvanId'].isin(irk_hayvanlar)]
        irk_artis = irk_gelisim[irk_gelisim['gunlukArtis'] > 0]['gunlukArtis']
        
        if len(irk_artis) > 0:
            ort_artis = irk_artis.mean()
            max_kilo = irk_gelisim['kilo'].max()
            print(f"   {irk}: {ort_artis:.3f} kg/gün (max: {max_kilo:.1f} kg)")
    
    # Sağlık durumu dağılımı
    print(f"\n🏥 SAĞLIK DURUMU DAĞILIMI:")
    saglik_dagilimi = df_gelisim['saglikDurumu'].value_counts()
    for durum, sayi in saglik_dagilimi.items():
        yuzde = (sayi / len(df_gelisim)) * 100
        print(f"   {durum}: {sayi} kayıt ({yuzde:.1f}%)")
    
    # Mevsimsel analiz
    print(f"\n🌤️ MEVSİMSEL PERFORMANS:")
    mevsim_performance = df_gelisim[df_gelisim['gunlukArtis'] > 0].groupby('mevsim')['gunlukArtis'].mean().sort_values(ascending=False)
    for mevsim, artis in mevsim_performance.items():
        print(f"   {mevsim}: {artis:.3f} kg/gün")
    
    return df_hayvanlar, df_gelisim

def prepare_ml_data():
    """
    Makine öğrenmesi için veri setini hazırlar
    """
    print("\n🤖 MAKİNE ÖĞRENMESİ VERİ HAZIRLIĞI")
    print("=" * 50)
    
    df_hayvanlar = load_hayvan_dataset()
    df_gelisim = load_gelisim_dataset()
    
    if df_hayvanlar is None or df_gelisim is None:
        return None, None
    
    # Tarih sütunlarını datetime'a çevir
    df_hayvanlar['dogumTarihi'] = pd.to_datetime(df_hayvanlar['dogumTarihi'])
    df_gelisim['tarih'] = pd.to_datetime(df_gelisim['tarih'])
    
    # Hayvan ve gelişim verilerini birleştir
    ml_data = df_gelisim.merge(df_hayvanlar[['id', 'irk', 'cinsiyet', 'dogumTarihi', 'dogumKilo']], 
                               left_on='hayvanId', right_on='id', how='left')
    
    # Feature engineering
    ml_data['yas_gun'] = (ml_data['tarih'] - ml_data['dogumTarihi']).dt.days
    ml_data['cinsiyet_encoded'] = ml_data['cinsiyet'].map({'Erkek': 1, 'Dişi': 0})
    
    # Irk one-hot encoding
    irk_dummies = pd.get_dummies(ml_data['irk'], prefix='irk')
    ml_data = pd.concat([ml_data, irk_dummies], axis=1)
    
    # Sağlık durumu encoding
    saglik_mapping = {'Hasta': 0, 'Zayıf': 1, 'Normal': 2, 'İyi': 3, 'Mükemmel': 4}
    ml_data['saglik_encoded'] = ml_data['saglikDurumu'].map(saglik_mapping)
    
    # Mevsim encoding
    mevsim_mapping = {'Kış': 0, 'İlkbahar': 1, 'Yaz': 2, 'Sonbahar': 3}
    ml_data['mevsim_encoded'] = ml_data['mevsim'].map(mevsim_mapping)
    
    # Sadece gerekli feature'ları seç
    feature_columns = [
        'yas_gun', 'cinsiyet_encoded', 'dogumKilo', 
        'boy', 'gogusEevresi', 'kalcaYuksekligi',
        'yemMiktari', 'sicaklik', 'nem',
        'saglik_encoded', 'mevsim_encoded'
    ] + [col for col in ml_data.columns if col.startswith('irk_')]
    
    # Hedef değişkenler
    target_columns = ['kilo', 'gunlukArtis']
    
    # Eksik değerleri temizle
    ml_data_clean = ml_data[feature_columns + target_columns + ['hayvanId', 'yasAy']].dropna()
    
    print(f"✅ ML veri seti hazırlandı:")
    print(f"   Toplam kayıt sayısı: {len(ml_data_clean)}")
    print(f"   Feature sayısı: {len(feature_columns)}")
    print(f"   Hedef değişken sayısı: {len(target_columns)}")
    
    return ml_data_clean, feature_columns, target_columns

if __name__ == "__main__":
    # Veri seti analizi yap
    df_hayvanlar, df_gelisim = analyze_dataset()
    
    if df_hayvanlar is not None:
        print("\n" + "="*50)
        
        # ML veri hazırlığı
        ml_data, features, targets = prepare_ml_data()
        
        if ml_data is not None:
            print(f"\n💾 ML veri seti dosyaya kaydediliyor...")
            ml_data.to_csv('ml_dataset.csv', index=False)
            print(f"✅ ml_dataset.csv kaydedildi")
            
            # Örnek tahmin için bazı kayıtları göster
            print(f"\n🔮 ÖRNEK KAYITLAR (İlk 5):")
            print(ml_data[['hayvanId', 'yasAy', 'kilo', 'gunlukArtis']].head()) 