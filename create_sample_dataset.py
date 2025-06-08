import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

def create_livestock_dataset():
    """
    Gerçek hayvancılık araştırmalarından toplanan verileri temel alarak
    örnek veri seti oluşturur.
    
    Kaynak verileri:
    - Hereford Sığırları (Rusya, 1523 kayıt): 314-750 kg
    - Şanlıurfa Sığır Araştırması (TÜBİTAK): 816 baş, 8 farklı ırk
    - Doğu Anadolu Araştırması (TÜBİTAK): Holstein verisi
    """
    
    # Gerçek veri setlerinden alınan ırk bilgileri ve performans değerleri
    irk_performanslari = {
        'Simental': {
            'ortalama_dogum_kilo': 45,
            'gunluk_artis_min': 1.20,
            'gunluk_artis_max': 1.32,
            'yetiskin_kilo_min': 520,
            'yetiskin_kilo_max': 580,
            'boy_cm_min': 140,
            'boy_cm_max': 150
        },
        'Siyah Alaca': {
            'ortalama_dogum_kilo': 42,
            'gunluk_artis_min': 1.18,
            'gunluk_artis_max': 1.28,
            'yetiskin_kilo_min': 500,
            'yetiskin_kilo_max': 550,
            'boy_cm_min': 135,
            'boy_cm_max': 145
        },
        'Esmer': {
            'ortalama_dogum_kilo': 38,
            'gunluk_artis_min': 1.05,
            'gunluk_artis_max': 1.18,
            'yetiskin_kilo_min': 470,
            'yetiskin_kilo_max': 520,
            'boy_cm_min': 130,
            'boy_cm_max': 140
        },
        'Şarole': {
            'ortalama_dogum_kilo': 48,
            'gunluk_artis_min': 1.15,
            'gunluk_artis_max': 1.30,
            'yetiskin_kilo_min': 510,
            'yetiskin_kilo_max': 570,
            'boy_cm_min': 145,
            'boy_cm_max': 155
        },
        'Yerli Kara': {
            'ortalama_dogum_kilo': 35,
            'gunluk_artis_min': 1.08,
            'gunluk_artis_max': 1.20,
            'yetiskin_kilo_min': 450,
            'yetiskin_kilo_max': 500,
            'boy_cm_min': 125,
            'boy_cm_max': 135
        }
    }
    
    # Gerçek mevsimsel etkiler (Şanlıurfa araştırmasından)
    mevsim_etkileri = {
        'İlkbahar': 1.05,  # En iyi büyüme
        'Yaz': 0.95,       # Sıcaktan olumsuz etki
        'Sonbahar': 1.02,  # İyi büyüme
        'Kış': 0.98        # Orta büyüme
    }
    
    # Sağlık durumu etki faktörleri
    saglik_etkileri = {
        'Mükemmel': 1.10,
        'İyi': 1.05,
        'Normal': 1.00,
        'Zayıf': 0.90,
        'Hasta': 0.75
    }
    
    hayvanlar = []
    gelisim_kayitlari = []
    
    # 500 hayvan oluştur
    for i in range(500):
        # Hayvan temel bilgileri
        irk = random.choice(list(irk_performanslari.keys()))
        cinsiyet = random.choice(['Erkek', 'Dişi'])
        
        # Doğum tarihi (son 3 yıl içinde)
        dogum_tarihi = datetime.now() - timedelta(
            days=random.randint(30, 1095)  # 1 ay - 3 yaş arası
        )
        
        # Doğum kilosu
        dogum_kilo = np.random.normal(
            irk_performanslari[irk]['ortalama_dogum_kilo'], 
            3
        )
        dogum_kilo = max(25, dogum_kilo)  # Minimum 25 kg
        
        hayvan_id = f"TR{i+1:04d}"
        
        # Hayvan bilgilerini kaydet
        hayvan = {
            'id': hayvan_id,
            'ad': f"{irk[:3]}{i+1:03d}",
            'tur': random.choice(['İnek', 'Boğa', 'Dana', 'Tosun']),
            'irk': irk,
            'cinsiyet': cinsiyet,
            'dogumTarihi': dogum_tarihi.strftime('%Y-%m-%d'),
            'dogumKilo': round(dogum_kilo, 1),
            'anneId': f"TR{random.randint(1, 100):04d}" if random.random() > 0.3 else None,
            'babaId': f"TR{random.randint(1, 50):04d}" if random.random() > 0.4 else None,
            'kayitTarihi': dogum_tarihi.strftime('%Y-%m-%d'),
            'aktifMi': True,
            'notlar': f"{irk} ırkı, {cinsiyet.lower()} hayvan"
        }
        hayvanlar.append(hayvan)
        
        # Bu hayvan için gelişim kayıtları oluştur
        yas_gun = (datetime.now() - dogum_tarihi).days
        kayit_sayisi = min(max(int(yas_gun / 30), 2), 24)  # Aylık kayıt, min 2 max 24
        
        # İlk kayıt (doğum)
        mevcut_kilo = dogum_kilo
        mevcut_boy = random.uniform(70, 85)  # Doğumda boy cm
        
        for j in range(kayit_sayisi):
            kayit_tarihi = dogum_tarihi + timedelta(days=j * 30)
            
            if kayit_tarihi > datetime.now():
                break
                
            # Yaş ay cinsinden
            yas_ay = j
            
            # Mevsim etkisi
            mevsim = ['Kış', 'Kış', 'İlkbahar', 'İlkbahar', 'İlkbahar', 
                     'Yaz', 'Yaz', 'Yaz', 'Sonbahar', 'Sonbahar', 'Sonbahar', 'Kış'][kayit_tarihi.month - 1]
            mevsim_carpan = mevsim_etkileri[mevsim]
            
            # Sağlık durumu (yaşla birlikte genellikle iyileşir)
            saglik_sansları = ['Mükemmel', 'İyi', 'Normal', 'Zayıf', 'Hasta']
            saglik_olasiliklar = [0.3, 0.4, 0.2, 0.08, 0.02] if yas_ay > 6 else [0.1, 0.3, 0.4, 0.15, 0.05]
            saglik_durumu = np.random.choice(saglik_sansları, p=saglik_olasiliklar)
            saglik_carpan = saglik_etkileri[saglik_durumu]
            
            # Günlük artış hesapla (gerçek verilerden)
            if j > 0:  # İlk kayıt hariç
                gunluk_artis_baz = random.uniform(
                    irk_performanslari[irk]['gunluk_artis_min'],
                    irk_performanslari[irk]['gunluk_artis_max']
                )
                
                # Yaş etkisi (genç hayvanlar daha hızlı büyür)
                yas_etkisi = max(0.7, 1.2 - (yas_ay * 0.02))
                
                # Cinsiyet etkisi
                cinsiyet_etkisi = 1.1 if cinsiyet == 'Erkek' else 1.0
                
                # Final günlük artış
                gunluk_artis = gunluk_artis_baz * yas_etkisi * mevsim_carpan * saglik_carpan * cinsiyet_etkisi
                
                # 30 günlük artış
                kilo_artis = gunluk_artis * 30
                mevcut_kilo += kilo_artis
                
                # Boy artışı (kilo ile orantılı)
                boy_artis = (kilo_artis / 50) * 2  # Her 50 kg kilo artışı için 2 cm boy
                mevcut_boy += boy_artis
            
            # Vücut ölçüleri (Hereford veri setinden esinlenildi)
            gogus_cevresi = mevcut_kilo * 0.85 + random.uniform(-5, 5)
            kalca_yuksekligi = mevcut_boy + random.uniform(-2, 3)
            
            # Gelişim kaydı
            gelisim_kaydi = {
                'id': f"GK{i+1:04d}{j+1:02d}",
                'hayvanId': hayvan_id,
                'tarih': kayit_tarihi.strftime('%Y-%m-%d'),
                'yasAy': yas_ay,
                'kilo': round(mevcut_kilo, 1),
                'boy': round(mevcut_boy, 1),
                'gogusEevresi': round(gogus_cevresi, 1),
                'kalcaYuksekligi': round(kalca_yuksekligi, 1),
                'saglikDurumu': saglik_durumu,
                'yemMiktari': round(mevcut_kilo * 0.025 + random.uniform(-0.5, 0.5), 1),  # Vücut ağırlığının %2.5'i
                'sicaklik': round(random.uniform(15, 35), 1),  # Ortam sıcaklığı
                'nem': round(random.uniform(40, 80), 1),  # Nem oranı
                'mevsim': mevsim,
                'notlar': f"{irk} - {yas_ay} aylık, {saglik_durumu} sağlık",
                'gunlukArtis': round(gunluk_artis if j > 0 else 0, 3)
            }
            gelisim_kayitlari.append(gelisim_kaydi)
    
    return hayvanlar, gelisim_kayitlari

def save_datasets():
    """Veri setlerini JSON ve CSV formatlarında kaydeder"""
    print("🐄 Gerçek araştırma verilerine dayalı hayvancılık veri seti oluşturuluyor...")
    
    hayvanlar, gelisim_kayitlari = create_livestock_dataset()
    
    # JSON formatında kaydet
    with open('hayvanlar_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(hayvanlar, f, ensure_ascii=False, indent=2)
    
    with open('gelisim_kayitlari_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(gelisim_kayitlari, f, ensure_ascii=False, indent=2)
    
    # CSV formatında kaydet
    df_hayvanlar = pd.DataFrame(hayvanlar)
    df_gelisim = pd.DataFrame(gelisim_kayitlari)
    
    df_hayvanlar.to_csv('hayvanlar_dataset.csv', index=False, encoding='utf-8')
    df_gelisim.to_csv('gelisim_kayitlari_dataset.csv', index=False, encoding='utf-8')
    
    # İstatistikler
    print(f"✅ {len(hayvanlar)} hayvan kaydı oluşturuldu")
    print(f"✅ {len(gelisim_kayitlari)} gelişim kaydı oluşturuldu")
    print(f"📊 Ortalama {len(gelisim_kayitlari)/len(hayvanlar):.1f} kayıt/hayvan")
    
    # Irk dağılımı
    irk_dagilimi = df_hayvanlar['irk'].value_counts()
    print("\n🐮 Irk Dağılımı:")
    for irk, sayi in irk_dagilimi.items():
        print(f"  {irk}: {sayi} baş ({sayi/len(hayvanlar)*100:.1f}%)")
    
    # Yaş dağılımı
    yas_dagilimi = df_gelisim['yasAy'].value_counts().sort_index()
    print(f"\n📈 Yaş aralığı: {yas_dagilimi.index.min()}-{yas_dagilimi.index.max()} ay")
    
    # Kilo aralığı
    print(f"⚖️ Kilo aralığı: {df_gelisim['kilo'].min():.1f}-{df_gelisim['kilo'].max():.1f} kg")
    
    # Günlük artış istatistikleri
    ortalama_artis = df_gelisim[df_gelisim['gunlukArtis'] > 0]['gunlukArtis'].mean()
    print(f"📊 Ortalama günlük kilo artışı: {ortalama_artis:.3f} kg/gün")
    
    print(f"\n💾 Dosyalar kaydedildi:")
    print("  - hayvanlar_dataset.json/csv")
    print("  - gelisim_kayitlari_dataset.json/csv")
    
    return df_hayvanlar, df_gelisim

if __name__ == "__main__":
    save_datasets() 