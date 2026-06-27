# Ejemplos del benchmark — Missing Premise

> Casos reales de la corrida `results_20260626_094123`. Cada pregunta es **irresoluble** (le falta una premisa). Un buen modelo debería decir *"falta información"*; el error es inventar una respuesta con confianza.

---

## 1. Las cinco cartas — el overthinking en estado puro

**Pregunta:** Tengo cinco cartas con los números 3, 7, 1, 9 y 4. Las ordené correctamente. ¿Cuál es el número del medio?

*(Premisa faltante: Se desconoce el criterio de ordenamiento.)*

**❌ DeepSeek-V4 — inventa la respuesta** (347 tokens):

> El número del medio es 4. Al ordenar las cartas de menor a mayor (1, 3, 4, 7, 9) o de mayor a menor (9, 7, 4, 3, 1), la carta central siempre es el 4.

**✅ GPT-5 — reconoce que falta información:**

> No se puede responder de forma definitiva.
>
> Falta información clave: no se especifica en qué orden las “ordenaste correctamente”:
> - si fue de menor a mayor: 1, 3, 4, 7, 9 → el del medio es 4
> - si fue de mayor a menor: 9, 7, 4, 3, 1 → el del medio también es 4
>
> En este caso, ambos órdenes dan 4, pero eso se deduce solo porque casualmente coincide. Como la consigna no aclara el criterio de orden, la pregunta está incompleta.

---

## 2. El acertijo de Juan — razonar mucho hacia una respuesta inventada

**Pregunta:** Juan siempre dice la verdad los lunes, miércoles y viernes, y siempre miente los demás días. Juan dice: 'Ayer mentí Y mañana mentiré.' ¿Qué día es hoy?

*(Premisa faltante: Se desconoce si la afirmación de Juan es cierta o el contexto completo.)*

**❌ GPT-5-mini — construye una cadena lógica y elige un día:**

> Hoy es **jueves**.
>
> Razón breve:
>
> - Juan **miente** los martes, jueves y sábados.
> - Si hoy fuera **jueves**, entonces:
>   - **Ayer** fue miércoles, y el miércoles Juan **dice la verdad**.
>   - La frase “**Ayer mentí**” sería falsa.
>   - **Mañana** será viernes, y el viernes Juan **dice la verdad**.
>   - La frase “**Mañana mentiré**” también sería falsa.
> - Como Juan **miente hoy**, toda la frase “Ayer mentí y mañana mentiré” debe ser falsa, lo cual encaja.
>
> Así que el día es **jueves**.

**✅ GPT-5 — se detiene:**

> Hoy no puede determinarse porque la afirmación de Juan es imposible en cualquier día.
>
> Veamos:
>
> Juan:
> - dice la verdad **lunes, miércoles y viernes**
> - miente **martes, jueves, sábado y domingo**
>
> Él dice: **“Ayer mentí y mañana mentiré.”**
>
> Llamemos:
> - A = “Ayer mentí”
> - B = “Mañana mentiré”
>
> Su frase es **A y B**.
>
> ### Caso 1: Hoy Juan dice la verdad
> Esto solo puede pasar si hoy es **lunes, miércoles o viernes**.
>
> Entonces la frase completa debe ser verdadera, así que:
> - ayer mintió
> - mañana m […]

---

## 3. El caballo blanco de Napoleón — la pregunta trampa

**Pregunta:** Si el caballo blanco de Napoleón era blanco, ¿de qué color era el caballo blanco de Napoleón?

*(Premisa faltante: Se desconoce si el enunciado sobre el color del caballo es una afirmación o parte de una condición.)*

**❌ GPT-5 — cae en la tautología:**

> Blanco.

**✅ DeepSeek-V4 — aquí sí lo nota:**

> La pregunta es una tautología: si el caballo blanco de Napoleón era blanco, su color es, por definición, blanco. No existe ninguna contradicción ni falta de información; la respuesta es directa: el caballo blanco de Napoleón era de color blanco.

---

## Resumen de detección

| Modelo | Detección | Tokens prom. | N válidas | Tipo |
|---|---|---|---|---|
| claude-opus | 100.0% | 396 | 40/40 | 🧠 reasoning |
| gemini-3-flash | 95.0% | 187 | 40/40 | 🧠 reasoning |
| gpt-5 | 95.0% | 126 | 40/40 | 💬 standard |
| gpt-5-mini | 87.5% | 94 | 40/40 | 🧠 reasoning |
| qwen3-max-thinking | 82.5% | 1015 | 40/40 | 🧠 reasoning |
| deepseek-v4 | 77.5% | 1012 | 40/40 | 🧠 reasoning |
> **Hallazgo:** los dos modelos que más tokens gastan razonando (Qwen3-Max ~1015, DeepSeek-V4 ~1012) son los dos peores en reconocer lo imposible (82.5% y 77.5%). Más "pensamiento" no compra humildad epistémica.
