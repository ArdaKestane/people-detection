import cv2
from ultralytics import YOLO
import numpy as np

# Modeli yükle
model = YOLO('yolov8m.pt')  # YOLOv8'in uygun bir modelini kullanın

# Video kaynağını aç
video_path = 'input.mov'  # Video dosyanızın yolu
cap = cv2.VideoCapture(video_path)

# Video yazıcıyı oluştur
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec olarak mp4v kullanılır, Mac'te uyumludur
output_path = 'output2.mp4'  # Çıktı video dosyasının yolu
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Sabit etiketler ve ID'ler
next_person_id = 1
id_mapping = {}  # ID'leri kişilere atamak için bir sözlük
id_counter = {}  # Kişilerin ID'lerini takip etmek için

def assign_id(box):
    """
    Bir kişiye ID atar. Kişinin mevcut ID'si varsa onu döner, yoksa yeni bir ID atar.
    """
    xyxy = box.xyxy[0].tolist()  # Kutu koordinatlarını al (x1, y1, x2, y2)
    x1, y1, x2, y2 = map(int, xyxy)
    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Kutunun merkezini hesapla

    # Kişi ID'lerini güncelle
    for person_id, (last_box_center, _) in id_counter.items():
        if np.linalg.norm(np.array(box_center) - np.array(last_box_center)) < 50:  # Yakınlık kontrolü
            # Eğer kişi yakınsa, eski ID'yi döndür
            id_counter[person_id] = (box_center, box)
            return person_id
    
    # Yeni bir ID atama
    global next_person_id
    id_counter[next_person_id] = (box_center, box)
    next_person_id += 1
    return next_person_id - 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # İnsanları tespit et
    results = model(frame)
    
    # Tespit edilen sonuçları işleme
    detected_ids = set()
    for result in results:  # Her bir sonuç üzerinde işlem yapma
        boxes = result.boxes  # Tespit edilen kutuları al
        
        for box in boxes:
            # Kutu bilgilerini al
            person_id = assign_id(box)
            xyxy = box.xyxy[0].tolist()  # Kutu koordinatlarını al (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Kutu çizme ve ID yazma
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person{person_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detected_ids.add(person_id)
    
    # Görünmeyen ID'leri serbest bırakma
    id_counter = {pid: data for pid, data in id_counter.items() if pid in detected_ids}
    
    # Çerçeveyi yaz
    out.write(frame)
    
    # Görüntüyü gösterme
    cv2.imshow('Frame', frame)
    
    # 'q' tuşuna basarak çıkma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()

# Bilgilendirme mesajı
print("İşlem tamamlandı! Video kaydedildi.")
