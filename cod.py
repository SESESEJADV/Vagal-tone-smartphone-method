import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import which
import moviepy.editor as mp
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import label
import pandas as pd

# =====================================
# 📁 RUTAS Y CONFIGURACIÓN DE CARPETAS
# =====================================
base = r"C:\Users\sebas\Desktop\Estudios y medidas\Estudio metodológico\ASR"
carpeta_videos = os.path.join(base, "videos")
carpeta_audios = os.path.join(base, "audios")
carpeta_resultados_resp = os.path.join(base, "resultados respiración")
carpeta_resultados_corazon = os.path.join(base, "resultados corazón")
carpeta_rr = os.path.join(base, "R-R")
carpeta_resultado_final = os.path.join(base, "resultado final")

# Crear carpetas principales
for carpeta in [carpeta_audios, carpeta_resultados_resp, carpeta_resultados_corazon, carpeta_rr, carpeta_resultado_final]:
    os.makedirs(carpeta, exist_ok=True)

extensiones_validas = ("*.mp4", "*.mov", "*.avi")

# ===============================
# 🔊 Configuración de FFmpeg
# ===============================
ffmpeg_path = r"C:\Users\sebas\Desktop\Respirar\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
AudioSegment.converter = which(ffmpeg_path)

# ===============================
# ❤️ FUNCIONES DE ANÁLISIS CARDÍACO
# ===============================
def extraer_canal_rojo(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    intensidad_rojo = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        r = frame[:, :, 2]
        intensidad_rojo.append(np.mean(r))

    cap.release()
    return np.array(intensidad_rojo), fps

def normalizar_señal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def filtrar_señal(signal, fps):
    lowcut, highcut = 0.7, 3.5
    nyquist = 0.5 * fps
    b, a = butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, signal)

def detectar_latidos(signal, fps):
    std = np.std(signal)
    mean = np.mean(signal)
    height = mean + 0.5 * std
    distance = int(fps * 0.4)
    prominence = std * 0.6
    peaks, _ = find_peaks(signal, height=height, distance=distance, prominence=prominence)
    return peaks

def graficar_segmento(original, filtrada, peaks, fps, carpeta_salida, idx, t_inicio):
    tiempo = np.arange(len(original)) / fps + t_inicio
    out_path = os.path.join(carpeta_salida, f"segmento_{idx+1}.png")

    plt.figure(figsize=(14, 6))
    plt.subplot(2,1,1)
    plt.plot(tiempo, original, color='gray')
    plt.title(f"Señal Original - Segmento {idx+1}")
    plt.xlabel("Tiempo (s)"); plt.ylabel("Intensidad")

    plt.subplot(2,1,2)
    plt.plot(tiempo, filtrada, color='red')
    plt.plot(tiempo[peaks], filtrada[peaks], "bx", label="Latidos")
    plt.title(f"Señal Filtrada con Latidos - Segmento {idx+1}")
    plt.xlabel("Tiempo (s)"); plt.ylabel("Amplitud")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def guardar_latidos(latidos_ms, carpeta_salida):
    path_out = os.path.join(carpeta_salida, "latidos.txt")
    with open(path_out, 'w') as f:
        for ms in latidos_ms:
            m = int(ms // 60000)
            s = int((ms % 60000) // 1000)
            ms_rest = int(ms % 1000)
            f.write(f"{m:02}:{s:02}.{ms_rest:03}\n")
    return path_out

def analizar_corazon(video_path, carpeta_salida):
    rojo, fps = extraer_canal_rojo(video_path)
    duracion = len(rojo) / fps
    segmentos = int(np.ceil(duracion / 30))
    latidos_totales = []

    for i in range(segmentos):
        i0 = int(i * 30 * fps)
        i1 = int(min((i + 1) * 30 * fps, len(rojo)))
        segmento = rojo[i0:i1]
        if len(segmento) < 10: continue

        norm = normalizar_señal(segmento)
        filtrada = filtrar_señal(norm, fps)
        peaks = detectar_latidos(filtrada, fps)
        ms_peaks = ((peaks + i0) / fps) * 1000
        latidos_totales.extend(ms_peaks)

        graficar_segmento(norm, filtrada, peaks, fps, carpeta_salida, i, t_inicio=i0/fps)

    return guardar_latidos(latidos_totales, carpeta_salida)

# ===============================
# 📏 CALCULAR INTERVALOS R-R
# ===============================
def time_to_seconds(t_str):
    m, s = t_str.split(":")
    return int(m) * 60 + float(s)

def calcular_rr(path_latidos, carpeta_salida):
    ruta_salida = os.path.join(carpeta_salida, "R-R.txt")
    with open(path_latidos, 'r') as f:
        lineas = [line.strip() for line in f if line.strip()]

    tiempos = [time_to_seconds(t) for t in lineas]
    with open(ruta_salida, 'w') as f:
        f.write("PAR DE LATIDOS | TIEMPO INICIO (s) | TIEMPO FIN (s) | INTERVALO R-R (s)\n")
        for i in range(1, len(tiempos)):
            ini, fin = tiempos[i - 1], tiempos[i]
            f.write(f"{i}-{i+1:<11} | {ini:.3f}             | {fin:.3f}          | {fin - ini:.3f}\n")
    return ruta_salida

# ===============================
# 🌬️ ANÁLISIS RESPIRACIÓN (INTERACTIVO)
# ===============================
def analizar_respiracion(video_path, carpeta_salida):
    import os
    import numpy as np
    from moviepy.editor import VideoFileClip
    from pydub import AudioSegment
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Cursor
    import tkinter as tk
    from tkinter import simpledialog
    from scipy.ndimage import label

    base = os.path.splitext(os.path.basename(video_path))[0]
    ruta_audio = os.path.join(carpeta_salida, f"{base}_audio.wav")
    salida_txt = os.path.join(carpeta_salida, f"{base}_ciclos.txt")
    salida_grafico = os.path.join(carpeta_salida, f"{base}_grafico.png")

    # Extraer audio
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(ruta_audio, verbose=False, logger=None)

    # Cargar y procesar audio
    audio = AudioSegment.from_file(ruta_audio).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)

    frame_ms = 50
    frame_size = int(audio.frame_rate * frame_ms / 1000)
    rms = [np.sqrt(np.mean(samples[i:i+frame_size]**2)) for i in range(0, len(samples), frame_size)]
    decibels = 20 * np.log10(np.array(rms) + 1e-6)
    decibels_smooth = np.convolve(decibels, np.ones(5)/5, mode='same')

    tiempos = np.arange(len(decibels_smooth)) * (frame_ms / 1000)

    # Mostrar gráfico interactivo
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tiempos, decibels_smooth, label="Decibeles suavizados")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (dB)")
    ax.set_title("Seleccione visualmente umbrales de apnea y respiración")
    cursor = Cursor(ax, useblit=True, color='black', linewidth=1, horizOn=True, vertOn=False)
    plt.show()

    # Pedir umbrales manualmente
    root = tk.Tk()
    root.withdraw()
    umbral_apnea = float(simpledialog.askstring("Umbral apnea", "Ingrese umbral de apnea (dB):", initialvalue="-40"))
    umbral_respiracion = float(simpledialog.askstring("Umbral respiración", "Ingrese umbral de respiración (dB):", initialvalue="-30"))

    # Clasificar fases
    fases = []
    for i, db in enumerate(decibels_smooth):
        t = i * (frame_ms / 1000)
        if db < umbral_apnea:
            fase = "Apnea"
        elif db >= umbral_respiracion:
            fase = "Respiracion"
        else:
            fase = "Intermedio"
        fases.append((t, fase))

    etiquetas = np.array([0 if f[1]=="Apnea" else 1 if f[1]=="Respiracion" else -1 for f in fases])
    bloques_intermedio, _ = label(etiquetas == -1)
    for b in range(1, bloques_intermedio.max()+1):
        idxs = np.where(bloques_intermedio == b)[0]
        if not len(idxs): continue
        antes = etiquetas[idxs[0]-1] if idxs[0] > 0 else etiquetas[idxs[-1]+1]
        despues = etiquetas[idxs[-1]+1] if idxs[-1]+1 < len(etiquetas) else antes
        etiquetas[idxs] = antes if antes == despues else despues

    etiqueta_fase = {0: "Apnea", 1: "Respiracion"}
    fases_limpias = [(fases[i][0], etiqueta_fase.get(etiquetas[i], "Apnea")) for i in range(len(fases))]

    nombres = ["Inhalacion", "Apnea 1", "Exhalacion", "Apnea 2"]
    ciclos, ciclo_actual = [], []

    i_inhal = next((i for i, f in enumerate(fases_limpias) if f[1] == "Respiracion"), None)
    if i_inhal is None:
        return

    fases_utiles = fases_limpias[i_inhal:]
    idx_fase, inicio = 0, fases_utiles[0][0]

    dur_min = 0.5
    for i in range(1, len(fases_utiles)):
        t, f = fases_utiles[i]
        if f != fases_utiles[i-1][1]:
            dur = t - inicio
            if dur >= dur_min:
                ciclo_actual.append((inicio, t, nombres[idx_fase]))
                if nombres[idx_fase] == "Apnea 2":
                    ciclos.append(ciclo_actual)
                    ciclo_actual, idx_fase = [], 0
                else:
                    idx_fase += 1
            inicio = t

    def fmt(seg):
        m, s = divmod(seg, 60)
        ms = (s - int(s)) * 1000
        return f"{int(m):02}:{int(s):02}.{int(ms):03}"

    with open(salida_txt, 'w') as f:
        for idx, ciclo in enumerate(ciclos, 1):
            f.write(f"Ciclo {idx}:\n")
            for ini, fin, nom in ciclo:
                f.write(f"  {nom}: {fmt(ini)} -> {fmt(fin)}\n")
            f.write("\n")

    # Guardar gráfico de barras
    colores = ['green' if f[1] == "Respiracion" else 'red' for f in fases_limpias]
    plt.figure(figsize=(12, 5))
    plt.bar(tiempos, decibels_smooth, width=0.05, color=colores, alpha=0.7)
    plt.axhline(umbral_apnea, color='black', linestyle='--', label=f'Umbral apnea ({umbral_apnea} dB)')
    plt.axhline(umbral_respiracion, color='blue', linestyle='--', label=f'Umbral respiración ({umbral_respiracion} dB)')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (dB)")
    plt.title("Ciclos respiratorios detectados")
    plt.legend()
    plt.tight_layout()
    plt.savefig(salida_grafico)
    plt.close()

    return salida_txt

# ===============================
# 📊 SINCRONIZAR R-R Y CICLOS
# ===============================
def sincronizar_rr_con_ciclos(path_ciclos, path_rr, excel_salida):
    def tiempo_a_segundos(t_str):
        if t_str == "N/A":
            return None
        m, s = t_str.split(":")
        return int(m) * 60 + float(s)

    with open(path_ciclos, 'r') as f:
        lineas_ciclos = [line.strip() for line in f if line.strip()]

    with open(path_rr, 'r') as f:
        lineas_rr = [line.strip() for line in f.readlines() if not line.startswith("PAR")]

    rr_tiempos = []
    for linea in lineas_rr:
        _, t_ini, t_fin, _ = linea.split("|")
        t_ini = float(t_ini.strip())
        t_fin = float(t_fin.strip())
        rr_tiempos.append((t_ini, t_fin))

    resultados = []
    ciclo_idx = None
    for line in lineas_ciclos:
        if line.startswith("Ciclo"):
            ciclo_idx = line.replace("Ciclo", "").replace(":", "").strip()
            continue
        if "->" not in line:
            continue

        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
        fase, tiempos = parts
        t_ini, t_fin = tiempos.strip().split("->")
        t_ini_s = tiempo_a_segundos(t_ini.strip())
        t_fin_s = tiempo_a_segundos(t_fin.strip())

        rr_en_fase = [rr for rr in rr_tiempos if rr[0] >= t_ini_s and rr[1] <= t_fin_s]
        promedio_rr = np.mean([rr[1] - rr[0] for rr in rr_en_fase]) if rr_en_fase else np.nan

        resultados.append({
            "Ciclo": ciclo_idx,
            "Fase": fase.strip(),
            "Inicio (s)": t_ini_s,
            "Fin (s)": t_fin_s,
            "Promedio R-R (s)": promedio_rr
        })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel(excel_salida, index=False)
    print(f"📄 Resultado guardado en: {excel_salida}")

# ===============================
# ▶️ EJECUCIÓN PRINCIPAL
# ===============================
print("🚀 Iniciando análisis completo de todos los videos...\n")

for ext in extensiones_validas:
    videos = glob.glob(os.path.join(carpeta_videos, ext))
    for video in videos:
        nombre_video = os.path.splitext(os.path.basename(video))[0]
        print(f"\n🎞️ Procesando: {nombre_video}")

        # Crear subcarpetas
        carpeta_resp = os.path.join(carpeta_resultados_resp, nombre_video)
        carpeta_corazon = os.path.join(carpeta_resultados_corazon, nombre_video)
        carpeta_rr_video = os.path.join(carpeta_rr, nombre_video)
        carpeta_final_video = os.path.join(carpeta_resultado_final, nombre_video)
        for carpeta in [carpeta_resp, carpeta_corazon, carpeta_rr_video, carpeta_final_video]:
            os.makedirs(carpeta, exist_ok=True)

        # Respiración y Corazón
        archivo_ciclos = analizar_respiracion(video, carpeta_resp)
        archivo_latidos = analizar_corazon(video, carpeta_corazon)
        archivo_rr = calcular_rr(archivo_latidos, carpeta_rr_video)

        # Sincronizar y exportar Excel
        excel_salida = os.path.join(carpeta_final_video, f"{nombre_video}_resumen.xlsx")
        sincronizar_rr_con_ciclos(archivo_ciclos, archivo_rr, excel_salida)

print("\n✅ Todos los análisis completados.")
