# =============================
# CONFIGURACIÓN Y DEPENDENCIAS
# =============================
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer, BadSignature
import os
from dotenv import load_dotenv
import json
import time
import threading
import queue
from google import genai
import asyncio

# Carga variables de entorno desde .env
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# =============================
# PROMPT PARA GEMINI
# =============================
PROMPT = """
# Rol del sistema:
Actuás como un generador experto de ejercicios de análisis de código en Python. Tenés una sólida formación en pedagogía, didáctica computacional y programación avanzada. Tu objetivo es generar preguntas tipo test (múltiple opción) para estudiantes universitarios de nivel intermedio. Cada pregunta debe fomentar el pensamiento lógico, la comprensión profunda de la ejecución del código y la interpretación de resultados no triviales.

# Objetivo:
Diseñar una pregunta de opción múltiple de análisis de código en Python, basada en un fragmento autocontenido de código desafiante, cumpliendo estrictos criterios de calidad, ejecución y formato.

# Especificaciones para evitar repetitividad y errores comunes:
- Prohibido generar ejercicios de recursividad donde el resultado correcto sea 120 (por ejemplo, factorial de 5). Si usas recursividad, el resultado debe ser diferente y el input debe variar en cada ejercicio.
- Si usas input(), el valor esperado no debe ser siempre 5 ni ningún valor fijo repetido. Varía los valores y asegúrate de que el enunciado lo indique claramente.
- No repitas estructuras de código, nombres de variables, ni patrones lógicos en preguntas consecutivas. Cada pregunta debe ser única y desafiante.
- Si usas listas, tuplas, diccionarios o conjuntos, varía su contenido y la lógica de manipulación en cada ejercicio.
- No generes preguntas donde la respuesta correcta sea siempre la misma en ejercicios similares.
- Si usas funciones recursivas, varía el tipo de problema (no solo factorial, suma, fibonacci, etc.).
- Si usas operaciones matemáticas, varía los operadores y los valores involucrados.
- Si usas cadenas, varía el tipo de manipulación (no solo invertir, concatenar, etc.).
- Si usas estructuras de control, varía el tipo de bucle, condición y lógica interna.
- No generes preguntas triviales ni con resultados evidentes.
- No generes preguntas donde la opción correcta no coincida exactamente con la salida real del código.
- Si el código tiene input(), asegúrate de que el valor esperado esté explícito en el enunciado y que la respuesta correcta corresponda a ese valor.
- No generes preguntas donde la explicación contradiga la opción correcta o corrija el resultado después de mostrar las opciones.
- Si detectas cualquier error en la generación, reinicia el proceso desde el paso 1.

# Estructura de generación (paso a paso):
1. **Generá un bloque de código Python autocontenido** que cumpla con los criterios detallados en la sección "Criterios del código". El código debe ser diferente a los generados anteriormente y evitar patrones repetitivos.
2. **Limitá el código a un máximo de 18 líneas ejecutables** (sin contar líneas en blanco ni comentarios), para evitar preguntas demasiado extensas y reducir el tamaño de la cookie de errores.
3. **Simulá mentalmente su ejecución** (o ejecutalo internamente) y determiná con exactitud su salida o el valor final de una variable clave.
4. **Redactá una pregunta clara**, basada en ese código, sin adornos ni ambigüedades. El enunciado debe estar contextualizado para análisis de código.
5. **Generá 4 opciones plausibles**, una de ellas correcta. Las incorrectas deben ser verosímiles.
6. **Verificá que la respuesta correcta coincida EXACTAMENTE con una de las opciones.**
7. **Escribí una explicación concisa y precisa**, enfocada en la lógica del código y la razón por la que la opción correcta es válida.
8. **Devolvé solo un objeto JSON válido**, con los campos especificados. No agregues texto adicional ni comentarios.

# Reglas estrictas de generación (no ignorar):
1. Antes de generar las opciones, SIMULÁ paso a paso la ejecución del código. Para cada línea, analizá el flujo, valores intermedios y salida.
2. La opción correcta DEBE COINCIDIR EXACTAMENTE con la salida real del código. Si no podés verificar esto mentalmente, no generes la pregunta.
3. No generes explicaciones que contradigan las opciones. Toda explicación debe confirmar directamente el valor de la respuesta correcta.
4. Si hay discrepancia entre el resultado del código y las opciones, REINICIÁ el proceso desde el punto 1. No te autocorrijas al final.

# Criterios del código:
- Sintaxis Python válida, compatible con versiones recientes.
- Nombres de variables y funciones en español, con estilo *camelCase*.
- Código indentado con 4 espacios (sin tabulaciones).
- Sin uso de librerías externas.
- Mínimo 8 líneas de código ejecutable y máximo 18 líneas ejecutables (sin contar comentarios ni líneas en blanco).
- Mínimo dos funciones definidas por el usuario, con parámetros y retorno.
- Al menos cinco bloques lógicos diferenciados (control de flujo, funciones, manipulación de estructuras de datos, etc.).
- Al menos una estructura de datos (lista, tupla, conjunto o diccionario) manipulada activamente.
- Uso de condicionales, bucles, llamadas a funciones encadenadas, y estructuras de control anidadas.
- Opcional: uso de `input()` con validación.
- Estilo de interpretación variable: análisis de salida o ejecución con datos específicos.
- Evitá repetir estructuras, patrones o lógicas de ejercicios típicos. Cada pregunta debe ser única y desafiante.

# Validación y control de calidad:
- Simulá el código paso a paso.
- Comprobá 3 veces que la salida y la opción correcta coinciden.
- Asegurate de que no existan ambigüedades, errores o trivialidades.
- Comprueba que los signos de comparación (==, !=, <, >) se usen correctamente en el contexto del código.
- Verificá que los valores de `input()` se encuentren en el enunciado de la pregunta.
- Asegura que todas las operaciones matemáticas y lógicas se realicen correctamente, considerando el tipo de datos (enteros, flotantes, cadenas).

# Formato de salida (obligatorio):
Devuelve únicamente un objeto JSON con esta estructura exacta:

{
  "Pregunta": "Texto claro, sin adornos. Enunciado técnico enfocado en la ejecución del código.",
  "Codigo": "Bloque de código Python autocontenido, bien indentado, formateado y funcional.",
  "Respuestas": ["Opción A", "Opción B", "Opción C", "Opción D"],
  "Respuesta correcta": "Debe coincidir exactamente con una de las opciones anteriores.",
  "Explicacion": "Explicación centrada en la ejecución paso a paso y en la lógica del código."
}

# Criterios de calidad y ejecución:
- Pregunta clara, sin ambigüedades ni adornos.
- Código autocontenido, ejecutable y con salida única.
- Opciones plausibles, una correcta y tres incorrectas pero verosímiles.
- Explicación precisa, enfocada en la lógica del código y la respuesta correcta.
- Asegura la correcta coincidencia entre la respuesta correcta y las opciones generadas.
- Los signos de comparación (==, !=, <, >) deben ser usados correctamente en el contexto del código.
- Verifica el verdadero peso de cada numero al comparar valores enteros y flotantes.
- Asegurate de seguir la estructura de generación paso a paso especificada. (obligatorio)

# Restricciones finales:
- Solo la salida JSON. No incluyas ningún texto adicional.
- Evitá preguntas redundantes, triviales o con valores repetidos.
- Fomentá variedad estructural en los códigos.
- Validación rigurosa antes de emitir la respuesta.
- El código generado no debe superar las 18 líneas ejecutables.
- Asegurate que los valores de los inputs() se encuentren en el enunciado de la pregunta.
- No generes la pregunta sin usar la simulación de ejecución del código.

# Prohibido:
- Generar salidas sin verificarlas.
- Producir preguntas con explicaciones que corrigen opciones incorrectas.
- Variar el formato. Solo el JSON especificado.
- Generar preguntas que no puedan ser verificadas por el modelo.
- Generar preguntas que no cumplan con los criterios de calidad y ejecución especificados.
"""

# =============================
# CLIENTE GENAI
# =============================
client = genai.Client(api_key=GENAI_API_KEY)

# =============================
# CACHE DE PREGUNTAS (COLA)
# =============================
CACHE_SIZE = 200  # Máximo de preguntas en cache
CACHE_MIN = 100   # Umbral mínimo para reponer el cache
pregunta_cache = queue.Queue(maxsize=CACHE_SIZE)

# =============================
# VALIDACIÓN DE PREGUNTAS
# =============================
def es_pregunta_valida(pregunta):
    """
    Verifica que la pregunta es válida y no contiene errores ni campos faltantes.
    """
    if not isinstance(pregunta, dict):
        return False
    if 'error' in pregunta:
        return False
    campos = ['pregunta', 'codigo', 'respuestas', 'respuesta_correcta']
    for campo in campos:
        if campo not in pregunta or not pregunta[campo]:
            return False
    if not isinstance(pregunta['respuestas'], list) or len(pregunta['respuestas']) != 4:
        return False
    return True

# =============================
# GENERACIÓN Y OBTENCIÓN DE PREGUNTAS
# =============================
def generar_pregunta():
    """
    Llama a Gemini para generar una pregunta nueva.
    Limpia el texto y lo convierte a un diccionario Python.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite", 
        contents=PROMPT
    )
    try:
        text = response.text.strip()
        # Limpia el texto de bloques de código Markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        pregunta_json = json.loads(text)
        # Validación de la estructura del JSON
        respuestas = pregunta_json.get("Respuestas")
        if isinstance(respuestas, str):
            respuestas = [r.strip() for r in respuestas.split(",")]
        elif not isinstance(respuestas, list):
            respuestas = []
        pregunta = {
            "pregunta": pregunta_json.get("Pregunta"),
            "codigo": pregunta_json.get("Codigo"),
            "respuestas": respuestas,
            "respuesta_correcta": pregunta_json.get("Respuesta correcta"),
            "explicacion": pregunta_json.get("Explicacion", "")
        }
        if not es_pregunta_valida(pregunta):
            return {"error": "Pregunta inválida o incompleta", "detalle": "Faltan campos o formato incorrecto", "texto": text}
        return pregunta
    except Exception as e:
        return {"error": "No se pudo extraer el JSON", "detalle": str(e), "texto": response.text}

# =============================
# HILO DE PRECARGA DE PREGUNTAS
# =============================
def precargar_preguntas():
    """
    Hilo en segundo plano que mantiene el cache de preguntas lleno.
    Solo consulta la API si el cache baja del umbral.
    """
    while True:
        if pregunta_cache.qsize() < CACHE_MIN:
            try:
                pregunta = generar_pregunta()
                # Solo la guarda si es válida
                if es_pregunta_valida(pregunta):
                    pregunta_cache.put(pregunta)
                time.sleep(5)  # Espera 5 segundos antes de volver a intentar
            except Exception as e:
                # Si es un error de cuota, espera más tiempo
                if "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(35)
                else:
                    time.sleep(5)
        else:
            time.sleep(2)  # Espera antes de volver a chequear

# Inicia el hilo de precarga al arrancar la app
threading.Thread(target=precargar_preguntas, daemon=True).start()

async def obtener_pregunta_cache_async():
    """
    Obtiene una pregunta del cache de forma no bloqueante para el event loop.
    Si el cache está vacío, genera una pregunta en caliente.
    """
    loop = asyncio.get_running_loop()
    try:
        pregunta = await loop.run_in_executor(None, lambda: pregunta_cache.get(timeout=10))
        if not es_pregunta_valida(pregunta):
            return generar_pregunta()
        return pregunta
    except Exception:
        pregunta = generar_pregunta()
        return pregunta

# =============================
# FASTAPI APP Y RUTAS
# =============================

# Inicialización de la app y sistema de plantillas
app = FastAPI()
templates_path = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=templates_path)

# Configuración de la clave secreta y serializador para cookies firmadas
SECRET_KEY = os.getenv("SESSION_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "SESSION_SECRET_KEY no está configurada. "
        "Establezca un valor seguro en su entorno o archivo .env."
    )
SESSION_COOKIE = "quiz_session"
serializer = URLSafeSerializer(SECRET_KEY)

def get_session(request: Request):
    """
    Recupera y valida la sesión del usuario desde la cookie.
    Si no existe o la firma es inválida, devuelve un dict vacío.
    """
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return {}
    try:
        return serializer.loads(cookie)
    except BadSignature:
        return {}

def set_session(response: Response, session_data: dict):
    """
    Serializa y firma los datos de sesión, y los guarda en la cookie de la respuesta.
    """
    cookie_value = serializer.dumps(session_data)
    response.set_cookie(SESSION_COOKIE, cookie_value, httponly=True, max_age=60*60*60)

def clear_session(response: Response):
    """
    Elimina la cookie de sesión.
    """
    response.delete_cookie(SESSION_COOKIE)

@app.get('/', name="inicio")
def inicio(request: Request):
    """
    Ruta de inicio: muestra la presentación y botón para comenzar el quiz.
    Limpia cualquier sesión previa.
    """
    response = templates.TemplateResponse('inicio.html', {'request': request})
    clear_session(response)
    return response

@app.get("/quiz", name="quiz")
async def quiz_get(request: Request):
    """
    Muestra la pregunta actual.
    Si la sesión no existe o está incompleta, la inicializa.
    """
    session = get_session(request)

    if not all(k in session for k in ['puntaje', 'total', 'inicio', 'pregunta_actual']) or session == {}:
        nueva_pregunta = await obtener_pregunta_cache_async()
        intentos = 0
        while not es_pregunta_valida(nueva_pregunta) and intentos < 10:
            await asyncio.sleep(2)
            nueva_pregunta = await obtener_pregunta_cache_async()
            intentos += 1
        if not es_pregunta_valida(nueva_pregunta):
            response = RedirectResponse(
                url=f'/error?detalle=Límite%20de%20intentos%20superado&texto=No%20se%20pudo%20generar%20una%20pregunta%20válida.%20Por%20favor%20intente%20nuevamente%20más%20tarde.',
                status_code=303
            )
            return response
        session = {
            'puntaje': 0,
            'total': 0,
            'inicio': int(time.time()),
            'pregunta_actual': nueva_pregunta,
            'errores': []
        }

    # Si la pregunta actual no es válida, reintenta obtener otra
    intentos = 0
    while not es_pregunta_valida(session['pregunta_actual']) and intentos < 10:
        await asyncio.sleep(2)
        session['pregunta_actual'] = await obtener_pregunta_cache_async()
        intentos += 1
    if not es_pregunta_valida(session['pregunta_actual']):
        response = RedirectResponse(
            url=f'/error?detalle=Límite%20de%20intentos%20superado&texto=No%20se%20pudo%20generar%20una%20pregunta%20válida.%20Por%20favor%20intente%20nuevamente%20más%20tarde.',
            status_code=303
        )
        return response

    pregunta = session['pregunta_actual']
    num_pregunta = session.get('total', 0) + 1
    response = templates.TemplateResponse(
        'quiz.html',
        {'request': request, 'pregunta': pregunta, 'num_pregunta': num_pregunta}
    )
    
    set_session(response, session)
    return response

@app.post('/quiz')
async def quiz_post(request: Request, respuesta: str = Form(...)):
    """
    Procesa la respuesta del usuario y muestra la siguiente pregunta o el resultado.
    Actualiza el puntaje y los errores en la sesión.
    Si se llega a 10 preguntas, redirige a la página de resultados.
    """
    session = get_session(request)
    if not all(k in session for k in ['puntaje', 'total', 'inicio', 'pregunta_actual']):
        return RedirectResponse(url='/', status_code=303)

    # Si la pregunta actual no es válida, reintenta obtener otra
    intentos = 0
    while not es_pregunta_valida(session['pregunta_actual']) and intentos < 10:
        await asyncio.sleep(2)
        session['pregunta_actual'] = await obtener_pregunta_cache_async()
        intentos += 1
    if not es_pregunta_valida(session['pregunta_actual']):
        response = RedirectResponse(
            url=f'/error?detalle=Límite%20de%20intentos%20superado&texto=No%20se%20pudo%20generar%20una%20pregunta%20válida.%20Por%20favor%20intente%20nuevamente%20más%20tarde.',
            status_code=303
        )
        return response

    seleccion = respuesta
    correcta = session['pregunta_actual']['respuesta_correcta']
    explicacion = session['pregunta_actual']['explicacion']
    session['total'] += 1

    if seleccion and seleccion.strip() == correcta.strip():
        session['puntaje'] += 1

    if session['total'] >= 10:
        tiempo = int(time.time() - session['inicio'])
        puntaje = session['puntaje']
        response = RedirectResponse(
            url=f'/resultado?correctas={puntaje}&tiempo={tiempo}',
            status_code=303
        )
        clear_session(response)
        return response

    # Si no ha terminado, obtiene una nueva pregunta y actualiza la sesión
    nueva_pregunta = await obtener_pregunta_cache_async()
    intentos = 0
    while not es_pregunta_valida(nueva_pregunta) and intentos < 10:
        await asyncio.sleep(2)
        nueva_pregunta = await obtener_pregunta_cache_async()
        intentos += 1
    if not es_pregunta_valida(nueva_pregunta):
        response = RedirectResponse(
            url=f'/error?detalle=Límite%20de%20intentos%20superado&texto=No%20se%20pudo%20generar%20una%20pregunta%20válida.%20Por%20favor%20intente%20nuevamente%20más%20tarde.',
            status_code=303
        )
        return response
    session['pregunta_actual'] = nueva_pregunta
    response = RedirectResponse(url='/quiz', status_code=303)
    set_session(response, session)
    return response

@app.get('/resultado')
def resultado(request: Request, correctas: int = 0, tiempo: int = 0):
    """
    Ruta para mostrar el resultado final.
    Recupera los errores desde la cookie temporal y los muestra junto al puntaje y tiempo.
    """
    # Recupera errores del localStorage usando JavaScript en resultado.html
    response = templates.TemplateResponse(
        'resultado.html',
        {'request': request, 'correctas': correctas, 'tiempo': tiempo, 'errores': []}  # errores vacío, se cargan en el frontend
    )
    return response

@app.get('/error')
def error(request: Request, detalle: str = '', texto: str = ''):
    """
    Ruta para mostrar errores personalizados.
    """
    return templates.TemplateResponse(
        'error.html',
        {'request': request, 'detalle': detalle, 'texto': texto},
        status_code=500
    )