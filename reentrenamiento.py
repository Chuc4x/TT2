import os
import base64
import requests
from io import BytesIO
from PIL import Image
from azure.storage.fileshare import ShareFileClient, ShareDirectoryClient
import uuid

# --- Configuración de Azure ---
AZURE_STORAGE_ACCOUNT_NAME = "dbilletesescom"
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")  # Cargado desde variables de entorno por seguridad
AZURE_FILE_SHARE_NAME = "dbilletesescom"
AZURE_FILE_DIRECTORY = "imagenes"

# --- Carpeta destino ---
base_path = "C:/Users/Jesus/Documents/GitHub/TT"
img_dir = os.path.join(base_path, "dataset", "imagenes", "train")
label_dir = os.path.join(base_path, "dataset", "labels", "train")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# --- Lista de candados esperados ---
candados_requeridos = {
    "banco", "hilo", "serie", "serie2", "numero",
    "texto", "benito", "patron", "marca", "carruaje"
}

# --- Conexión al recurso compartido ---
directory_client = ShareDirectoryClient(
    account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.file.core.windows.net/",
    share_name=AZURE_FILE_SHARE_NAME,
    directory_path=AZURE_FILE_DIRECTORY,
    credential=AZURE_STORAGE_ACCOUNT_KEY
)

files = directory_client.list_directories_and_files()
total = 0
guardadas = 0

for f in files:
    if not f.name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    total += 1
    print(f"\n📥 Procesando: {f.name}")

    try:
        # Descargar imagen desde Azure
        file_client = ShareFileClient.from_connection_string(
            conn_str=f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT_NAME};AccountKey={AZURE_STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net",
            share_name=AZURE_FILE_SHARE_NAME,
            file_path=f"{AZURE_FILE_DIRECTORY}/{f.name}"
        )
        img_bytes = file_client.download_file().readall()

        # Enviar a la API
        response = requests.post(
            "http://127.0.0.1:8000/api/subir-imagen/",  # o tu URL de producción
            files={"imagen": (f.name, img_bytes, "image/jpeg")}
        )

        if response.status_code != 201:
            print(f"❌ Error con {f.name}: {response.status_code}")
            continue

        data = response.json()
        detecciones = data.get("detecciones", [])
        candados_detectados = {d["candado"] for d in detecciones}

        if candados_detectados != candados_requeridos:
            print(f"⚠️ Candados incompletos en {f.name}, descartada.")
            continue

        # Guardar imagen decodificada
        img_base64 = data["imagen_base64"]
        image_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(image_data)).convert("RGB")
        nombre_base = str(uuid.uuid4())
        image_path = os.path.join(img_dir, f"{nombre_base}.jpg")
        label_path = os.path.join(label_dir, f"{nombre_base}.txt")
        img.save(image_path)

        # Guardar etiquetas formato YOLO
        w, h = img.size
        with open(label_path, "w") as f_txt:
            for d in detecciones:
                x1, y1, x2, y2 = d["coordenadas"]
                try:
                    class_id = list(candados_requeridos).index(d["candado"])
                except ValueError:
                    continue  # por si hay un candado no esperado
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f_txt.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        guardadas += 1
        print(f"✅ Guardada: {nombre_base}.jpg y .txt")

    except Exception as e:
        print(f"❌ Fallo en {f.name}: {e}")

print(f"\n📊 Resultado final: {guardadas}/{total} imágenes procesadas correctamente.")
