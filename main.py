# =============================
# CONFIGURACIÓN Y DEPENDENCIAS
# =============================
from fastapi import FastAPI, Request, Form, Response, Cookie
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

# Carga variables de entorno desde .env
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# =============================
# PROMPT PARA GEMINI
# =============================
PROMT = """
Eres un generador experto de ejercicios de análisis de código en Python, dirigidos a estudiantes universitarios. Tu objetivo es crear preguntas de opción múltiple que exploren en profundidad el razonamiento lógico, la comprensión estructural y la ejecución paso a paso de programas reales. Cada pregunta debe contener:

- Un enunciado claro, directo y técnico.

- Un bloque de código Python autocontenible, complejo, extenso y bien formateado.

- Cuatro opciones de respuesta.

- Una única respuesta correcta.

- Una explicación breve basada en la lógica interna del programa.

OBJETIVO
- No generes ejercicios triviales. Cada pregunta debe ser una situación que obligue al estudiante a leer, analizar y simular la ejecución del código paso por paso, combinando múltiples conceptos en simultáneo.

REQUISITOS GENERALES DE CÓDIGO
Lenguaje y estilo:

- Python válido, sintaxis de la última versión estable.

- Nombres de variables y funciones en español, usando notación camelCase.

- Usar 4 espacios por nivel de indentación (no tabs).

- El código debe ser autocontenible, sin librerías externas.

- Estructura obligatoria del código:

- Mínimo 5 bloques lógicos distintos (ej. definiciones, condicionales, bucles, estructuras de datos, etc.).

- Usar al menos dos funciones definidas por el usuario.

- Incorporar estructuras de control de flujo anidadas.

- Incluir al menos una estructura de datos (lista, tupla, diccionario o set) y manipularla en el código.

- Incluir, cuando sea relevante, llamadas recursivas o interacción entre funciones.

- Usar casos reales de acumulación, filtrado, ordenamiento o validación de datos.

- Cuando se incluya input(), deben procesarse y transformarse los datos de entrada (por ejemplo: convertir a enteros, separar cadenas, recorrerlos).

- Siempre que sea posible, combinar lógica condicional + iterativa + funciones + estructuras de datos en un mismo bloque.

- Temas a integrar y combinar en una misma pregunta:

- Estructuras condicionales: if, elif, else

- Estructuras repetitivas: for, while, incluyendo control de iteración (ej. break, continue)

- Funciones: def, con múltiples argumentos y retorno de valores

- Listas, tuplas, diccionarios y sets, incluyendo acceso, modificación y recorrido

- Recursividad, si se justifica

- Operaciones con strings y procesamiento de texto (cuando sea útil)

- Anidamiento de estructuras y efectos colaterales entre funciones

- Mínimo 8 líneas de código funcional, sin contar espacios ni declaraciones triviales

VARIEDAD Y ESTILO DE PREGUNTAS
- Selecciona aleatoriamente uno de los siguientes tipos:

- "¿Qué salida tendrá el siguiente código?"

- "¿Qué salida tendrá el siguiente código si se ingresan los siguientes valores?"
(si elegís esta opción, el enunciado debe indicar claramente los valores a ingresar, y el código debe usar input() adecuadamente)
(asegúrate de que los valores de input() sean únicos (valores enteros, flotantes o cadenas), puedes basarte en fechas importantes)

- "¿Qué valor tendrá la variable X al finalizar la ejecución del siguiente código?"

RESTRICCIONES Y CONTROL DE CALIDAD
- Evitá repeticiones de código o lógica entre preguntas.

- No generes preguntas genéricas, predecibles ni de resolución inmediata.

- Cuando generes preguntas que incluyan input(), debés:

    - Usar valores de entrada distintos en cada pregunta generada, evitando repeticiones de combinaciones numéricas o de texto.

    - Asegurarte de que los valores de entrada sean variados, no triviales ni predecibles, e idealmente elegidos aleatoriamente dentro de un rango significativo (por ejemplo: entre 0 y 100, o usar listas de nombres, strings, etc.).

    - No reutilices combinaciones previas como [5, 1, 3] o [7, 2, 1]. Las entradas deben ser significativamente diferentes entre preguntas.

    - Si se ingresan múltiples valores, deben tener propósitos distintos dentro del código (por ejemplo: índice, cantidad, valor a comparar), y deben afectar el resultado de forma no lineal.

    - Evitá que la entrada sea solo una excusa para ejecutar un print. Debe tener un impacto lógico real en la ejecución.

- El código debe ser desafiante pero comprensible, idealmente similar al que se encontraría en una evaluación universitaria real.

- Antes de determinar la respuesta correcta, simulá mentalmente el código paso por paso. No asumas.

- La respuesta correcta debe coincidir exactamente con una de las opciones.

- No incluyas comentarios, explicaciones, bloques Markdown ni texto adicional fuera del JSON.

Formato de salida (único objeto JSON):

{
  "Pregunta": "Texto de la pregunta clara y concisa.",
  "Codigo": "Fragmento de código Python válido, autocontenible y bien formateado.",
  "Respuestas": ["Respuesta A", "Respuesta B", "Respuesta C", "Respuesta D"],
  "Respuesta correcta": "Respuesta correcta exactamente igual a una de las opciones",
  "Explicacion": "Explicación breve y general del razonamiento que lleva al resultado correcto, sin hacer referencia a letras o posiciones de respuesta."
}

IMPORTANTE:
- No incluyas absolutamente ningún texto fuera del objeto JSON.

- No uses etiquetas, explicaciones, ni bloques Markdown.

- El resultado debe ser solo el objeto JSON bien formado.
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
        model="gemini-2.0-flash", 
        contents=PROMT
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

def obtener_pregunta_cache():
    """
    Obtiene una pregunta del cache (espera hasta 10s).
    Si el cache está vacío, genera una pregunta en caliente.
    """
    try:
        pregunta = pregunta_cache.get(timeout=10)
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

def set_errores_cookie(response: Response, errores: list):
    """
    Serializa y guarda la lista de errores en una cookie separada, solo para mostrar en resultados.
    """
    # Limita el tamaño serializando solo los campos esenciales
    errores_livianos = []
    for err in errores:
        errores_livianos.append({
            'pregunta': err.get('pregunta', ''),
            'codigo': err.get('codigo', ''),
            'respuesta_correcta': err.get('respuesta_correcta', ''),
            'respuesta_usuario': err.get('respuesta_usuario', ''),
            'explicacion': err.get('explicacion', '')
        })
    cookie_value = serializer.dumps(errores_livianos)
    response.set_cookie("quiz_errores", cookie_value, max_age=60*60*10)

def get_errores_cookie(request: Request):
    """
    Recupera la lista de errores desde la cookie separada.
    """
    quiz_errores = request.cookies.get("quiz_errores")
    if not quiz_errores:
        return []
    try:
        return serializer.loads(quiz_errores)
    except BadSignature:
        return []

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

    if not all(k in session for k in ['puntaje', 'total', 'inicio', 'pregunta_actual', 'errores']) or session == {}:
        nueva_pregunta = obtener_pregunta_cache()
        intentos = 0
        while not es_pregunta_valida(nueva_pregunta) and intentos < 10:
            time.sleep(2)
            nueva_pregunta = obtener_pregunta_cache()
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
        time.sleep(2)
        session['pregunta_actual'] = obtener_pregunta_cache()
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

    # Recupera errores de la cookie (si existen)
    errores = get_errores_cookie(request)

    # Si la pregunta actual no es válida, reintenta obtener otra
    intentos = 0
    while not es_pregunta_valida(session['pregunta_actual']) and intentos < 10:
        time.sleep(2)
        session['pregunta_actual'] = obtener_pregunta_cache()
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
    else:
        errores.append({
            'pregunta': session['pregunta_actual']['pregunta'],
            'codigo': session['pregunta_actual']['codigo'],
            'respuesta_correcta': correcta,
            'explicacion': explicacion,
            'respuesta_usuario': seleccion
        })

    if session['total'] >= 10:
        tiempo = int(time.time() - session['inicio'])
        puntaje = session['puntaje']
        response = RedirectResponse(
            url=f'/resultado?correctas={puntaje}&tiempo={tiempo}',
            status_code=303
        )
        clear_session(response)
        set_errores_cookie(response, errores)
        return response

    # Si no ha terminado, obtiene una nueva pregunta y actualiza la sesión
    nueva_pregunta = obtener_pregunta_cache()
    intentos = 0
    while not es_pregunta_valida(nueva_pregunta) and intentos < 10:
        time.sleep(2)
        nueva_pregunta = obtener_pregunta_cache()
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
    set_errores_cookie(response, errores)
    return response

@app.get('/resultado')
def resultado(request: Request, correctas: int = 0, tiempo: int = 0, quiz_errores: str = Cookie(default=None)):
    """
    Ruta para mostrar el resultado final.
    Recupera los errores desde la cookie temporal y los muestra junto al puntaje y tiempo.
    """
    errores = get_errores_cookie(request)
    response = templates.TemplateResponse(
        'resultado.html',
        {'request': request, 'correctas': correctas, 'tiempo': tiempo, 'errores': errores}
    )
    response.delete_cookie("quiz_errores")
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