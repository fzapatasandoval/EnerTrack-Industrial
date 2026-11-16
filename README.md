# EnerTrack Industrial

Prototipo académico de inteligencia artificial aplicada a procesos industriales.

El pronóstico Holt-Winters, Google Sheets y Gemini ya funcionan en un workflow de n8n. Ese flujo no se modifica en este repositorio. El código Python de aquí se integrará más adelante con n8n.

Hay un clasificador visual, una búsqueda de reportes y un pronóstico diario de consumo. La integración con n8n queda para más adelante.

## Entorno

Pensado para macOS con Apple Silicon, incluido un MacBook Air M4.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-modelos.txt
```

`requirements.txt` es el que usa Streamlit Community Cloud y solo instala el dashboard. `requirements-modelos.txt` conserva PyTorch y el resto del entrenamiento local.

## Dataset de defectos superficiales

Las imágenes de esta etapa provienen de la NEU Surface Defect Database, publicada por Kechen Song y Yunhui Yan (Northeastern University). Se obtuvo la copia pública de Kaggle `kaustubhdikshit/neu-surface-defect-database` con `kagglehub`, sin credenciales.

Esa copia está empaquetada como NEU-DET. Trae las imágenes en carpetas `train` y `validation` (240 y 60 por clase) y anotaciones XML de detección. Para clasificación se reúnen todas las imágenes por categoría. El reparto original y los XML no se usan.

La planta EnerTrack de este proyecto es ficticia. Estas fotos no salen de esa planta. El dataset no incluye una clase `normal`, y no se agregó ninguna.

En la copia descargada, `patches/patches_101.jpg` y `patches/patches_105.jpg` tienen el mismo contenido. El sorteo estratificado, con semilla 42, usa 70 % entrenamiento, 15 % validación y 15 % prueba. Después, esas dos copias quedan juntas en entrenamiento. El resultado guardado es 1.261 imágenes de entrenamiento, 269 de validación y 270 de prueba. En `patches` eso queda en 211, 44 y 45. Las otras cinco clases quedan en 210, 45 y 45.

El manifiesto está en `results/neu_surface_defects_split.csv`, con la ruta relativa al proyecto, la etiqueta y la partición. Las imágenes quedan en `data/neu_surface_defects/` y no se versionan, salvo `crazing/crazing_110.jpg`, que el dashboard muestra.

```bash
python src/prepare_neu_dataset.py
```

La segunda ejecución reutiliza la caché de `kagglehub` y no vuelve a descargar el archivo.

## Clasificación de defectos

Las seis clases originales se mantienen, en este orden de índices:

0. `crazing`
1. `inclusion`
2. `patches`
3. `pitted_surface`
4. `rolled-in_scale`
5. `scratches`

La correspondencia queda en `models/label_map.json`. Todas las imágenes son defectos. El dataset no tiene clase normal y el modelo tampoco: una superficie sin defecto se vería obligada a caer en una de estas seis clases.

El entrenamiento lee únicamente `results/neu_surface_defects_split.csv`. No se volvió a descargar el dataset ni se cambió la partición: 1.261 entrenamiento, 269 validación y 270 prueba. Las 270 imágenes de prueba se usaron solo para la evaluación final.

### Redes

ResNet18 y MobileNetV3 Small salen de torchvision. Los pesos que sí se descargaron y verificaron con `check_hash` son:

- `ResNet18_Weights.IMAGENET1K_V1`, archivo `resnet18-f37072fd.pth` (46.830.571 bytes), 1.000 clases ImageNet.
- `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, archivo `mobilenet_v3_small-047dcff4.pth` (10.306.551 bytes), 1.000 clases ImageNet.

La última capa lineal se sustituyó por una salida de seis clases. El resto del modelo quedó congelado, en modo evaluación, para no actualizar la normalización por lotes. Solo se entrenó esa capa final: 3.078 parámetros en ResNet18 y 6.150 en MobileNetV3 Small. No hubo fine-tuning completo. Adam con tasa 0.001, lotes de 32, semilla 42 y MPS. Máximo de 8 épocas y parada temprana con paciencia 2 si el macro F1 de validación no mejora. El checkpoint guardado es el de mejor macro F1 de validación: época 5 para MobileNet y época 4 para ResNet18.

En entrenamiento, las imágenes se redimensionan a 256, reciben un volteo horizontal y una rotación de hasta 10 grados, y después un recorte central de 224. En validación y prueba no hay aumentos: redimensionado a 256, recorte central de 224 y la normalización de ImageNet (media 0.485, 0.456, 0.406; desviación 0.229, 0.224, 0.225).

### Referencia simple

Sobre las imágenes en gris se calcularon media, desviación, mínimo, máximo, percentiles 10, 50 y 90, un histograma de 16 intervalos y la media y desviación de las diferencias horizontales y verticales. Un `StandardScaler` y una regresión logística se ajustaron solo con entrenamiento. El parámetro `C` se eligió entre 0.01, 0.1, 1 y 10 según el macro F1 de validación. Ganó `C=10`, el valor más alto de esa grilla. La prueba no participó en esa elección.

### Resultados en las 270 imágenes de prueba

| Modelo | Accuracy | Macro F1 | Aciertos | ms por imagen | Probabilidad media de la clase predicha |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV3 Small | 0.9926 | 0.9926 | 268/270 | 9.93 | 0.930 |
| Estadísticos y regresión logística | 0.9778 | 0.9777 | 264/270 | 1.74 | 0.975 |
| ResNet18 | 0.9704 | 0.9703 | 262/270 | 9.30 | 0.867 |

La referencia simple supera a ResNet18 en accuracy y macro F1, y además es más rápida. MobileNetV3 Small queda por encima de las dos. El tiempo incluye abrir la imagen y preparar la entrada; en las redes también incluye el forward en MPS. La probabilidad de la referencia es la de `predict_proba`, no una confianza softmax.

MobileNetV3 Small solo se equivocó dos veces: `inclusion` predicha como `pitted_surface`. Las otras cinco clases quedaron en 45/45.

ResNet18 concentró los fallos en `scratches`: 5 de 45 se predijeron como `inclusion`. También marcó 2 `inclusion` como `pitted_surface` y 1 `patches` como `crazing`.

La referencia falló sobre todo en `scratches`: 3 como `inclusion` y 1 como `pitted_surface`. Además, 1 `inclusion` pasó a `pitted_surface` y 1 `pitted_surface` a `scratches`.

Las matrices completas están en `results/confusion_matrix_mobilenet_v3_small.csv`, `results/confusion_matrix_resnet18.csv` y `results/confusion_matrix_estadisticos_textura_logreg.csv`. Precision y recall por clase están en `results/per_class_metrics.csv`.

Para una integración posterior con n8n, el modelo a usar es MobileNetV3 Small: es el más preciso y unos 10 ms por imagen siguen siendo poco para un flujo. La referencia es la alternativa más simple y rápida si no se quiere cargar PyTorch. ResNet18 no aporta ni precisión ni velocidad frente a esa referencia.

```bash
python src/train_classifiers.py
python src/predict_defect.py ruta/de/imagen.jpg --model mobilenet_v3_small
```

### Limitaciones

- El resultado describe la copia NEU usada aquí, no la planta ficticia EnerTrack.
- Sin clase normal, el clasificador no puede decir que una pieza está bien.
- El softmax no está calibrado como probabilidad de acierto.
- ResNet18 se detuvo al dejar de mejorar en validación; no se descongeló la red porque MobileNet ya estaba en 0.9926 de macro F1 de validación.
- `C=10` quedó en el borde de la grilla prevista. No se amplió esa búsqueda.
- La partición de prueba tiene 45 imágenes por clase. Dos errores cambian el accuracy en 0.74 puntos.
- MPS y los aumentos aleatorios hacen que otra ejecución pueda moverse un poco, aunque la semilla es 42.

## Búsqueda de reportes de mantenimiento

No apareció un corpus público en español, descargable y referido a reportes de una planta de envases plásticos. Las tesis sobre plantas de plástico no publican ese historial, y el conjunto sintético `Jvachier/industrial-maintenance-synthetic` está en inglés y no describe ese proceso. Por eso los textos de esta etapa son simulados.

Hay 48 reportes en `data/maintenance_reports/reportes.json`, marcados como `simulado`. Hablan de una planta ficticia de envases plásticos: motores, consumo, vibraciones, temperatura y mantenimiento preventivo. No son registros de una empresa. La fuente queda repetida en cada resultado para que Gemini pueda citar el identificador `SIM-xxx` sin tratarlo como un documento real.

Las 16 consultas están en `data/maintenance_reports/consultas.json`. Los reportes relevantes se fijaron al escribir cada incidente, antes de ejecutar el modelo.

El Transformer es `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, multilingüe y sin fine-tuning. La biblioteca instalada es `sentence-transformers` 5.1.2. Los embeddings se normalizan y se ordenan por similitud coseno. En esta ejecución el dispositivo fue MPS. La alternativa no usa Transformer: TF-IDF sobre el texto de los reportes y coseno con la consulta. Las dos devuelven tres reportes, con identificador, texto, puntaje, fuente, carácter simulado y cita.

Recall@3 es la proporción de relevantes que entran en esos tres. MRR es el promedio de 1 dividido por el puesto del primer relevante en el ranking completo.

| Método | Recall@3 | MRR |
| --- | ---: | ---: |
| Búsqueda por palabras (TF-IDF) | 0.9375 | 0.96875 |
| Embeddings y similitud coseno | 0.78125 | 0.88542 |

En este conjunto la búsqueda por palabras recupera más relevantes dentro del top 3. El Transformer acierta cuando la consulta parafrasea el reporte, como en Q11: «botellas deformadas por el agua de proceso caliente» devuelve `SIM-026` (0.720) y `SIM-025` (0.706), que hablan de alabeo y agua a 29 °C. También falla por cercanía temática. En Q01 el primer resultado es `SIM-022` (0.677), un engrase de mezclador, y el seguimiento real `SIM-002` queda fuera del top 3. En Q15 pone por delante el pico de demanda de los molinos (`SIM-017`, 0.460) y deja el atasco del molino M-9 en el tercer puesto. En Q10 los dos métodos dejan fuera `SIM-024`; las palabras colocan primero el agua de proceso (`SIM-025`) y el Transformer coloca primero la alarma del molde (`SIM-023`, 0.839).

```bash
python src/evaluate_maintenance_search.py
python src/search_reports.py "consulta en español" --metodo semantico
```

### Limitaciones de la búsqueda

- El corpus es pequeño, sintético y está escrito con el vocabulario del mismo escenario. Por eso las palabras compartidas favorecen a TF-IDF.
- No hubo fine-tuning ni se usó Gemini para ordenar o para decidir qué reporte era relevante.
- Un acierto en este escenario no dice que el modelo sirva sobre órdenes reales de otra planta.

## Pronóstico de consumo

La serie es la hoja `Historico_90_dias` del archivo local `EnerTrack_Datos_Certamen2.xlsx`. Se copió sin modificar a `data/consumo/historico_90_dias.csv`: 90 días consecutivos, del 2026-08-11 al 2026-11-08, columna `consumo_kwh`. No hay fechas inválidas, duplicados ni huecos, así que no se interpoló ningún día. Todos los registros dicen «Operación habitual simulada».

La prueba son los últimos 21 días, del 2026-10-19 al 2026-11-08, igual que el nodo original. Cada pronóstico usa solo información hasta el día anterior: evaluación rolling de un paso.

Holt-Winters es el aditivo de periodo 7 del nodo Python, con la misma inicialización y la misma grilla. El SSE para elegir parámetros se calcula desde el día 14 hasta el día anterior a la prueba. En esta serie quedaron `alpha=0.1`, `beta=0.1` y `gamma=0.3`. El nodo original informa MAE redondeado a dos decimales y no calcula RMSE; aquí el RMSE se agrega con los errores sin redondear.

La LSTM tiene una sola capa de 16 unidades y una ventana de 14 días, porque la estacionalidad del proyecto es semanal y dos semanas permiten ver el ciclo anterior. Hay 14 días de validación, justo antes de la prueba, para la parada temprana. La media y la desviación salen solo del tramo inicial. Semilla 42, MPS, máximo 50 épocas y paciencia 8. La validación empeoró desde la época 1, así que el checkpoint conservado es ese. Esos 14 días de validación sí entran en el SSE de Holt-Winters, porque el nodo original no separa un tramo de validación.

| Modelo | MAE (kWh) | RMSE (kWh) | ms por pronóstico |
| --- | ---: | ---: | ---: |
| Holt-Winters | 60.16 | 73.13 | 0.00093 |
| Ingenuo, día anterior | 437.67 | 629.59 | 0.00006 |
| LSTM | 444.09 | 533.10 | 1.15 |

Holt-Winters sigue el ciclo semanal: los sábados y domingos bajan de unos 2.300 kWh a cerca de 900–1.500 kWh. Su mayor error en la prueba es el viernes 2026-10-23: 2.019 kWh reales frente a 2.209 kWh pronosticados. El modelo ingenuo acierta entre semana y falla en el cambio de domingo a lunes; el 2026-10-26 pronostica 917 kWh y el consumo fue 2.351 kWh. La LSTM se quedó alrededor de 2.030 kWh todos los días, así que el error se concentra en los fines de semana. En MAE queda un poco por detrás del ingenuo; en RMSE queda por delante, porque no comete los saltos extremos del día anterior. No se cambió el entrenamiento para invertir ese resultado.

Para el prototipo académico conviene seguir con Holt-Winters: es el más preciso, es el método que ya usa n8n y el pronóstico de un día es inmediato.

```bash
python src/import_historico.py
python src/train_lstm_forecast.py
python src/predict_consumption.py
```

## Análisis integrado

`src/analyze_integrated.py` reúne el pronóstico, la imagen y la búsqueda sin reentrenar y sin llamar a Gemini. Los consumos y los reportes son sintéticos. Las imágenes NEU son públicas y reales. Esas tres fuentes no corresponden a un mismo evento industrial.

Las métricas de MAE, accuracy y Recall@3 que aparecen en el JSON salen de los archivos ya guardados en `results/`. Describen las evaluaciones anteriores, no la consulta que se acaba de enviar.

n8n Cloud no puede cargar PyTorch ni MiniLM dentro de un nodo. La API local escucha solo en `127.0.0.1` y todavía no se publica. El workflow original no se modifica.

El secreto no se guarda en el repositorio. En la terminal, antes de arrancar:

```bash
export ENERTRACK_API_TOKEN='un-valor-largo-elegido-por-ti'
.venv/bin/python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

Se detiene con Control-C en esa misma terminal. `GET /health` no pide clave. `POST /analyze` exige la cabecera `X-API-Key` con el mismo valor. El cuerpo lleva `registros` (`fecha`, `consumo_kwh`), `imagen` como ruta relativa dentro de `data/neu_surface_defects` y `consulta`. No admite rutas absolutas ni `../`.

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ENERTRACK_API_TOKEN" \
  -d '{"registros":[{"fecha":"2026-08-11","consumo_kwh":2346}],"imagen":"crazing/crazing_110.jpg","consulta":"consumo del compresor"}'
```

El ejemplo de `curl` necesita al menos 35 registros; la prueba local usa los 90 del CSV.

```bash
python src/analyze_integrated.py \
  --csv data/consumo/historico_90_dias.csv \
  --imagen data/neu_surface_defects/crazing/crazing_110.jpg \
  --consulta "¿Por qué aumentó el consumo eléctrico del compresor de aire?" \
  --salida results/integrated_sample.json
```

## Dashboard

`dashboard.py` muestra el gráfico histórico, el filtro de fechas y las métricas de evaluación ya guardadas. Esas métricas no son el último análisis del workflow. La fotografía NEU y los reportes `SIM-xxx` también son el ejemplo local del 2026-11-08.

El último análisis sale de una sola fila de `Registro_alertas`: la ejecución con `fecha_ejecucion` más reciente. Las decisiones del supervisor salen de `Registro_decisiones`. Si Google Sheets no está conectado, el panel lo indica y muestra el ejemplo local con su fecha. No lo presenta como información en tiempo real.

En local, sin la hoja:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run dashboard.py
```

Para conectar las dos hojas con la misma cuenta de servicio:

1. En Google Cloud, crea una cuenta de servicio y descarga su JSON. Déjalo fuera del repositorio, por ejemplo en `$HOME/credenciales/enertrack-sheets.json`.
2. Activa la API de Google Sheets en ese proyecto.
3. Abre el libro `EnerTrack_Datos_Certamen2` y compártelo con el `client_email` del JSON, con permiso de lector. El libro debe contener `Registro_alertas` y `Registro_decisiones`.
4. En la misma terminal donde arranca Streamlit:

```bash
export ENERTRACK_SHEETS_CREDENTIALS="$HOME/credenciales/enertrack-sheets.json"
export ENERTRACK_SPREADSHEET_ID='1V2im7Ar9qfM5oQL_WF3URSqJKj4c2DdptzpYy67FDh4'
.venv/bin/python -m streamlit run dashboard.py
```

`ENERTRACK_SPREADSHEET_ID` puede omitirse: el valor de arriba es el libro del examen. No copies el JSON al proyecto ni a `.streamlit/secrets.toml` si ese archivo puede quedar en el repositorio. `.streamlit/secrets.toml` está ignorado.

En Streamlit Community Cloud el archivo principal es `dashboard.py`. En Settings, Secrets, pega el JSON como TOML. No subas el archivo de la cuenta:

```toml
spreadsheet_id = "1V2im7Ar9qfM5oQL_WF3URSqJKj4c2DdptzpYy67FDh4"

[gcp_service_account]
type = "service_account"
project_id = "el-proyecto-de-la-cuenta"
private_key_id = "el-id-del-json"
private_key = "la-clave-del-json-con-saltos-\\n"
client_email = "la-cuenta@el-proyecto.iam.gserviceaccount.com"
client_id = "el-id-de-cliente"
token_uri = "https://oauth2.googleapis.com/token"
```

En la nube, `private_key` conserva los saltos de línea del JSON como `\n`. Cloud instala solo `requirements.txt`. El panel lee `st.secrets["gcp_service_account"]` y `st.secrets["spreadsheet_id"]`; en local sigue valiendo la variable de entorno. No se publica hasta revisar las credenciales.

## Estructura

- `data/neu_surface_defects/`: imágenes NEU. Solo se versiona `crazing/crazing_110.jpg`.
- `data/maintenance_reports/`: reportes y consultas simulados.
- `data/consumo/`: serie diaria exportada de Historico_90_dias.
- `models/`: checkpoints de imágenes y de la LSTM. No se versionan los pesos.
- `dashboard.py`: panel Streamlit. No ejecuta los modelos.
- `src/`: preparación, clasificación, búsqueda y pronóstico.
- `results/`: métricas de imágenes, de búsqueda y de pronóstico.
