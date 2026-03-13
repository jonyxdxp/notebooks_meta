

# 1. Crear la instancia

# Entra a lambdalabs.com → Cloud → Launch Instance
# Elige A10 (24GB) — suficiente para tus modelos y más barato que A100
# Región: us-east-1 o us-west-2 (más disponibilidad)
# Agrega tu SSH key antes de lanzar (Settings → SSH Keys)


# 2. conectarte:

ssh ubuntu@<ip-address>





# 3. Setup del entorno (una sola vez)

# Lambda ya viene con CUDA, PyTorch y conda preinstalados — verificar versiones
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Clonar tu repo
git clone https://github.com/jonyxdxp/notebooks_meta.git
cd notebooks_meta

# Instalar dependencias
pip install -r requirements.txt





# 4. Migrar los datos de Google Drive

# Instalar rclone
curl https://rclone.org/install.sh | sudo bash

# Configurar Google Drive
rclone config
# → sigue el wizard: n → name "gdrive" → Google Drive → browser auth

# Copiar checkpoints y cache al servidor
rclone copy gdrive:checkpoints /home/ubuntu/checkpoints
rclone copy gdrive:data/cache  /home/ubuntu/data/cache




# 5. Adaptar los paths del CFG

CFG = _C(dict(
    ...
    cache_dir  = '/home/ubuntu/data/cache',
    ckpt_dir   = '/home/ubuntu/checkpoints',
    raw_data_dir = '/home/ubuntu/data/dailydialog_raw',
    ...
))




# 6. Correr el entrenamiento

# En lugar de celdas de Colab, convierte cada stage a un script .py
# (ya los tenemos — son los archivos que generamos)

# Correr Stage 1
python train_text_jepa.py

# Con nohup para que siga corriendo si te desconectas
nohup python train_text_jepa.py > logs/stage1.log 2>&1 &
tail -f logs/stage1.log





# 7. Guardar resultados antes de terminar la instancia

# Siempre antes de apagar — Lambda no tiene storage persistente por defecto
rclone copy /home/ubuntu/checkpoints gdrive:checkpoints
rclone copy /home/ubuntu/data/cache  gdrive:data/cache