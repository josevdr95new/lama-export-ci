# Exportar LaMa a ONNX (1024 / 1536) con GitHub Actions

Convierte los pesos originales de big-lama en modelos ONNX de alta resolución
para LamaLite. Corre en servidores de GitHub (rápido, no usa tu PC).

## Pasos

1. **Crea un repositorio nuevo en GitHub** (público es gratis e ilimitado)
   por ejemplo `lama-export-ci`.

2. **Sube esta carpeta** al repo (con GitHub Desktop: File → Add local repository
   → selecciona `lama-export-ci` → Publish). Debe subir:
   - `.github/workflows/export-lama.yml`
   - `export_lama.py`
   - este README

3. **Ejecuta el workflow**: en la web del repo → pestaña **Actions** →
   "Export LaMa a ONNX" → **Run workflow** → deja "1024 1536" → **Run**.

4. Espera ~10-20 minutos. Al terminar, entra en la ejecución y abajo en
   **Artifacts** descarga `lama-onnx`.

5. **Copia los `.onnx`** a:
   `C:\Users\josev\Documents\GitHub\LamaLite\models\`
   (nombres: `lama_1024.onnx`, `lama_1536.onnx`)

6. Reinicia LamaLite — el backend los detecta solo y los usa automáticamente
   para recortes grandes (calidad nativa tipo IOPaint).

## Notas

- Public repo = Actions gratis e ilimitado.
- El workflow clona `Carve-Photos/lama` y descarga los pesos `big-lama` de
  HuggingFace por sí solo: no subas pesos al repo.
- Para otras resoluciones: Run workflow con otro valor (ej. `768`).
