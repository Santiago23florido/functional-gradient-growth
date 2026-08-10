# Estado: crecimiento hacia la mejor arquitectura con el mínimo de parámetros

Rama `perf/phase1-tangent-system-reuse`. Nunca llevar a `main`. No borrar ni
commitear `cifar_phase1.log` ni `cifar_run.log` (sin trackear, del usuario).

## Objetivo

Media ≥ **0.925** con ≤ **600 parámetros** en N=1024, 4 semillas
(`model_seed` 0-3, `train_seed` fijo en 0). **No alcanzado en media** (0.900).
Dos semillas ya lo cumplen: 0.941 @ 602p y 0.925 @ 629p.

Y una restricción del usuario que aún no está resuelta: **el método no debe
sesgar hacia una arquitectura fija en todas las bases de datos** (como hacía con
la uniforme). Debe encontrar la mejor durante el desarrollo.

## Intocable (dicho explícitamente por el usuario)

- Certificado `eps < min(threshold, 0.5)` estricto, sobre la sonda de train.
- Lema 3.5 y su intervalo de tasa.
- `family_order: [tangent]` — **no añadir familias**. Solo valen las que
  certifican exactamente.
- El `lr` interno del ladder (`certify_family_inner_learning_rate: 0.01`) —
  vetado por riesgo a las certificaciones.
- Preservación de función en todo crecimiento.
- Las optimizaciones de rendimiento ya hechas.
- Nada de constantes ajustadas a este dataset. Nada de topes manuales de
  parámetros como solución (sí como instrumento de medida).

## Comandos

```bash
cd /home/santiago/StageFrugal/stable-tiny
PYTHONPATH=src python -m stable_tiny --config <copia.yaml> --no-wandb --no-plot
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/ -q
# baseline: 482 pasan, 3 fallan en test_regularized_mlp.py (preexistentes)
# ruff baseline: 137 (los B023 nuevos son el mismo falso positivo que ya había x50)
```

### Configs de trabajo: en `runs/`, NO en `/tmp`

`runs/cfg/base600_s{0..3}.yaml` (baseline, tope 600 como instrumento) y
`runs/cfg/up600_s{0..3}.yaml` (baseline + barrido ascendente). `runs/` está en
`.gitignore`. Se reconstruyen desde `configs/fgd/family_ladder_N1024.yaml` con:

```bash
for s in 0 1 2 3; do
  sed -e "s/^  model_seed: 0$/  model_seed: ${s}/" \
      -e 's/^  growth_selection: unified_expansion$/  growth_selection: unified_expansion\n  growth_where: certified_gain\n  growth_where_joint: true\n  certify_adaptive_growth: true/' \
      -e 's/^  max_total_parameters:$/  max_total_parameters: 600/' \
      configs/fgd/family_ladder_N1024.yaml > runs/cfg/base600_s${s}.yaml
  sed 's/^  certify_family_functional_lr: 1.0$/  certify_family_functional_lr: 1.0\n  certify_family_functional_lrs: [1.0, 2.0, 4.0, 8.0]/' \
      runs/cfg/base600_s${s}.yaml > runs/cfg/up600_s${s}.yaml
done
```

### ⚠️ CORRER DE UNA EN UNA

Lanzar 8-10 corridas en paralelo **agotó la RAM y bloqueó la máquina**, y el
reinicio borró `/tmp` entero: se perdieron todas las configs y **todos los logs
de las dos sesiones**. Los números sobrevivieron solo porque estaban escritos
aquí y en los mensajes de commit. Una corrida sola tarda ~2 min y usa ~4 GB de
23; en paralelo tardaban 9 min cada una, así que secuencial **no es más lento en
total**. No hay ninguna razón para paralelizar.

## Commits de esta sesión

- `dbb9d5a` puntuar la neurona despierta (ω=0 anula sus columnas del jacobiano)
- `a7382b3` despertar en direcciones **distintas** (el mismo escalar las dejaba
  colineales: rango 1 en vez de k)
- `f3d53f0` poda por tolerancia — dispara **0 veces**, no hay capacidad muerta
- `882f00d` **movimientos conjuntos** + ledger `[WHY]` por candidato
- `80e8255` `growth_where_probe` seleccionable (train | validation | both)

## Lo medido (no re-litigar)

| configuración | media | params |
|---|---|---|
| conjuntos, libre | **0.9200** | 956 |
| conjuntos, tope 600 | 0.9000 | 608 |
| ranking sobre validación | 0.9087 | 1188 |
| ranking sobre ambas sondas | 0.8925 | 1009 |
| referencia `rank_ceiling` | 0.918 | 671 |

**Mejor convencional (AdamW, grid de lr por forma, 3 semillas):**
`4-19-13-13-1` 551p → **0.9250** · `4-13-22-10-1` 614p → 0.9285 ·
`4-16-17-17-1` (la que crece el método) 693p → **0.9135, la peor de seis**.

⚠️ El barrido de 371 arquitecturas que dio "techo 0.888" es **inválido**:
entrenó todas con `lr=1e-2`. Con grid por forma el techo real es 0.9285.

**Refutado con medición válida:** poda (0 disparos), mudanza de ráfaga
(0.392 en una semilla), compra en toda localización competitiva, `min_cosine`
0.9→0.866 (no-op exacto), `inner_steps` `[16,64]`→`[16,64,256]` (no-op exacto —
ambos configuran `parametric_gd`, que **no está en `family_order`**).

**La sonda determina la arquitectura** (hallazgo firme): rankear sobre
validación produce capa central ancha en las 8 corridas (19-31-18, 17-26-16,
19-27-19, 21-27-21; ratios 1.19-1.72) contra 1.06-1.16 sobre train. Pero no
mejora la media y sube params.

## El bloqueante, validado de las líneas del log

`s1` se hunde a 0.83 con **las tres sondas**. Causa medida:

| | familia certifica | pasos tangentes rechazados | train_loss |
|---|---|---|---|
| s0 (0.941) | **12** | 25 | 0.0009 |
| s1 (0.830) | **2** | 25 | 0.0033 |

Mismos rechazos, 6× menos certificaciones de familia. **Subajusta** (train_loss
más alta, no más baja). Es independiente del *dónde*: su arquitectura es casi
idéntica a la de s0 con el mismo presupuesto. Las otras tres semillas promedian
0.93, así que **todo el déficit contra el objetivo es esta semilla**.


## LO QUE FUNCIONA HOY (`fe4c88d`) — leer esto primero

`growth_where: expressivity_bottleneck` en los tres configs de `configs/fgd/`.
UNA neurona en el argmax de `activation_gradient * sum(eigenvalues_extension^2)`
— el termino de extension de TINY, sin su `parameter_update_decrease`.

**N=1024, 4 semillas, <=600 params, 70 epochs, a igualdad de computo:**

| | media | formas |
|---|---|---|
| `certified_gain` (anterior) | 0.9230 | 15-16-15 en 3 de 4 |
| **`expressivity_bottleneck`** | **0.9435** | 4 distintas, ratios 1.46-2.44 |

Las 4 superan 0.925 individualmente. Reproducido exacto en las 4. **El pipeline
es determinista** (verificado: re-ejecutar da el mismo resultado bit a bit), asi
que cualquier diferencia entre corridas es causal, nunca ruido.

### RESUELTA: como parar sin presupuesto (ver `docs/budget_free_stopping.md`)

El candidato vivo de abajo esta **implementado y medido**:
`growth_bottleneck_crossfold_folds` (0 = apagado y bit a bit identico).
Ajusta la extension en K-1 pliegues, congela `(alpha, omega)` y la puntua
contra la `N` del pliegue retenido. La identidad
`<alpha omega, N> = sum(s_i^2)` hace que eso sea **el mismo cuello de botella**,
medido donde la direccion no se eligio. Normalizado por las dos normas de
Frobenius es un coseno, asi que la magnitud se cancela y la regla transfiere.
Bar: `t = media(b) / (desv(b)/sqrt(K)) > 1`.

**Dimensiona a la tarea, que era la pregunta** (presupuesto libre, config
canonica, solo cambian los knobs de datos):

| corrida | params | widths | acc | crecimientos rechazados |
|---|---|---|---|---|
| facil, off | 270 | 6-9-16 | 0.9996 | — |
| **facil, K=5** | **158** | 5-6-12 | **1.0000** | **8 epochs** |
| dificil, off | 252 | 11-8-10 | 0.1581 | — |
| **dificil, K=5** | **252** | 11-8-10 | 0.1581 | **0** |

La dificil es **identica**: el test corrio en cada decision y no rechazo
ninguna (`t` 4.3-5.9, todos los pliegues positivos). La facil pierde el 41% de
los parametros y la accuracy SUBE; una vez resuelta la tarea `t` llega a
**-9.13**, o sea que la direccion que propone el cuello de botella dentro de
muestra **contradice** los datos que no vio. Esa asimetria es lo que un tope no
puede hacer, porque un tope no sabe que tarea esta cortando.

**4 semillas N=1024, presupuesto libre, pareadas:**

| | s0 | s1 | s2 | s3 | media |
|---|---|---|---|---|---|
| off acc | 0.9352 | 0.8101 | 0.5691 | 0.8237 | 0.7845 |
| **on acc** | 0.9282 | 0.7964 | 0.5691 | 0.8237 | **0.7794** |
| off params | 529 | 280 | 158 | 283 | 312.5 |
| **on params** | **378** | 286 | 158 | 283 | **276.3** |

-11.6% de parametros por -0.005 de accuracy, y **s2 y s3 son bit a bit
identicas** — el criterio corrio en cada decision y no rechazo nada. Que sea
inerte donde no pasa nada es la mitad de lo que se le pide.

⚠️ **Leer esa columna de accuracy en corto.** El config canonico son 25 epochs
y las cuatro semillas alcanzan su mejor accuracy en la ULTIMA epoch: estan
limitadas por computo, no por estructura. No es comparable contra el 0.9435 del
headline, que fue a 70 epochs con tope 600.

⚠️ **Y una afirmacion mia que la medicion corrigio:** dije "decide SI crecer,
nunca DONDE". Falso. Poner a cero es por capa, asi que anular el argmax le cede
el turno a la segunda: s1 **no rechazo ni un crecimiento** y aun asi aterrizo en
`4-9-20` en vez de `7-11-12`, con 22 eventos contra 19. Lo que si se sostiene es
que ningun valor superviviente se reescala, asi que el ranking entre los que
pasan es el de dentro de muestra.

**Coste: +9.7% de reloj a K=5** (145 s -> 159 s, N=1024 s0), no el 10x que
sugiere la aritmetica de `2K` pasadas: `growth_crossfold_seconds` marca 21.0 s
pero la corrida solo se alarga 14 s, porque rechazar crecimiento tambien AHORRA
(`where_total_seconds` 4.48 -> 2.27 con el mismo `tangent_system_total`).
**El coste de MNIST no es este** — alli la sonda son 704x784 o 10000 filas y hay
que re-medirlo antes de fiarse.

Bug destapado y arreglado de paso: `growth_where_no_bottleneck` se pasaba a
`fallback()` desde `fe4c88d` sin estar registrado en `PROFILE_FIELDS`, asi que
`FGD_PROFILE=1` reventaba con KeyError **la primera vez que el crecimiento
paraba de verdad** — justo el evento que ese contador existe para registrar.

### EL PROBLEMA INVERSO: en MNIST no paraba, es que no arrancaba

Distinto del de arriba y hay que no confundirlos. Aquel es "deja de crecer
cuando ya no hace falta"; este es "empieza a crecer cuando si hace falta".

MEDIDO, `mnist-depth3-small-seed-0`, 150 epochs terminadas:

```
ep=1  eps=0.8527  P=1612          por encima de la barra -> crece
ep=1  eps=0.4965  P=2399          dos crecimientos lo meten debajo
ep=2  eps=0.8505  P=2399          vuelve a subir -> crece
ep=2  eps=0.4975  P=2405  (+6)    certifica... y ya nunca mas
```

**2 crecimientos, 16 pasos externos en 150 epochs, accuracy 0.109**, y todos
los certificados sanos. eps se aparca dos milesimas bajo la barra; el Lema 3.5
da `eta_bar = 0.0025` ahi y el paso comprometido fue `1.55e-05`, asi que la
perdida se mueve un 0.005% por paso, eps no sube, y no se vuelve a crecer.
Estable: 1500 epochs darian lo mismo.

El segundo crecimiento fueron **6 parametros en una capa interna**, y con eso
basto para bajar eps de 0.85 a 0.4975. Ese es el fondo: **`eps < 1/2` se
satisface con una red trivialmente pequeña** en MNIST, porque eps es relativo.
El certificado hace su trabajo — garantiza que existe un paso — pero no es un
indicador de capacidad, y el crecimiento estaba cableado a el.

**N1024 no lo sufre** porque alli eps vive POR ENCIMA de 0.5 (41/46, 29/44,
28/30, 32/44 de los pasos en las cuatro semillas), asi que la puerta dispara
sola.

**El defecto, en una linea.** `certify.py:723` es
`while (epsilon >= target or forced_remaining > 0)`, y el lookahead que si mira
esa banda (`_growth_reduces_lookahead_epsilon`, `pipeline.py:2251`) solo sabia
RETORNAR: un "no" salia, un "si" caia al `while`, que lo excluye. **Era un freno
sin acelerador.**

**Arreglado** con `certify_growth_lookahead_entry`, que deja al mismo predicado
autorizar la entrada. Acotado por construccion: la autorizacion la consume la
primera iteracion y el `break` de `growth_turn_taken` que ya existia cierra el
bucle, o sea **una neurona por paso externo y solo mientras crecer le gane a
entrenar**. Puentea `_growth_pays` igual que `is_forced`, porque bajo el
certificado el hueco `(eps - target)` es NEGATIVO y esa regla no significa nada
ahi.

⚠️ **Restringido a `family_order: [matrix_free_tangent]` por el validador**, y
esa restriccion es el punto: la ruta del tangente y `family_ladder_N1024.yaml`
son invariantes, y "el flag esta apagado alli" no es garantia — alguien lo
enciende y se acabo. Es un error de carga. Verificado ademas por regresion:
N1024 semilla 0 reproduce **`acc 0.9352 / 529 params / widths (10,18,14) / 23
crecimientos` identico** tras el cambio.

**Sin medir todavia en MNIST.** El riesgo es que el lookahead diga que si
siempre: con el entrenamiento parado el clon quieto casi no mejora y crecer gana
por defecto. El diagnostico es el par de contadores
`certify_growth_warranted_entries` / `certify_growth_not_warranted`: **si el
primero sube monotono y el segundo se queda en 0, no esta discriminando** y hay
que atacar el clon quieto, no aflojar el freno.

### El planteamiento original (conservado)

Nada de esto se detiene solo. Las 4 semillas paran al agotar los 600
parametros; s1 llego al tope en la epoch 35 y paso las 35 restantes congelada
(34 de 35 epochs con `model update rejected`, no entrenando). El tope es
delicado porque **nunca se sabe cuanto es** de antemano.

La senal EXISTE y es limpia: el cuello de botella colapsa **diez ordenes de
magnitud** (4.9e-02 -> 1e-10..1e-12) justo cuando la tarea queda resuelta.
Lo que falta es leerla como orden de parar.

**Descartado con medicion:**
- Umbral absoluto sobre los valores propios (`tiny_statistical_threshold`):
  depende de la escala de cada base. Vetado.
- Marchenko-Pastur: a `gamma = r/n ~ 0.015` (r = ancho de capa 2-32, n = 1024)
  el borde queda en **1.26x el bulk** y no filtra nada. MP necesita `r/n`
  apreciable; aqui el regimen asintotico no aplica. Sin tope la forma degenera
  a `20-32-2` y el crecimiento revienta su tolerancia numerica.
- Criterio de primer orden (comparar contra `parameter_update_decrease`):
  **invalido bajo function-preserving**, porque al crecer `f` no se mueve y no
  se cobra ningun decremento.

**Candidato vivo (YA IMPLEMENTADO, ver arriba): validacion cruzada en K
pliegues.** Ajustar la direccion de
extension en K-1 pliegues de la sonda y medir su cuello de botella en el
retenido. Una direccion real da valor positivo fuera de muestra; el ruido
fluctua alrededor de cero. `t = media(b_k) / (desv(b_k)/sqrt(K))`, crecer solo
si `t > 1`. **No depende de que `r/n` sea grande**, que es lo que mato a MP, y
el "umbral" es estadistico y no una magnitud, asi que transfiere entre bases.
Cuesta K pasadas de estadisticos por capa y por evento.

### Las dos vias de crecimiento (no confundirlas)

| | pregunta | criterio | % de neuronas |
|---|---|---|---|
| via 1 `grow_until_certified` | que crecimiento hace que EXISTA un paso | argmin exacto de eps | 12-36% |
| via 2 bloque por epoch | donde falta expresividad | **cuello de botella** | 64-88% |

La via 1 **NO es sustituible**: su argmin de eps es lo que garantiza que eps
baje, que es lo que hace terminar su bucle (tiene teorema) y lo que un paso
necesita para existir. Sustituirla por el cuello de botella dio media 0.7380
con s1 paralizada en 46 params. Y las dos **discrepan** sobre donde falta
capacidad: ratear el conteo adaptativo por cuello de botella dio CERO disparos
en 4 semillas.

## El barrido de η: hecho, medido, y FUNCIONA hacia arriba (`d843873`)

Implementado como `certify_family_functional_lrs` (lista; `()` = comportamiento
anterior bit a bit). Cada candidata certifica por su propio
`RelErr(Δ,r) < min(threshold, 0.5)` sobre la misma sonda, se para en la primera
que certifica, y **η=1 va primero**, así que una corrida que certifica ahí es
idéntica a hoy y al mismo coste: la escalera sube **solo donde antes se crecía**.
6 tests nuevos; suite en baseline (482/3); ruff en 137.

**La dirección del barrido es lo que decide.**

| configuración (tope 600) | s0 | s1 | s2 | s3 | media |
|---|---|---|---|---|---|
| baseline (reproducido exacto) | 0.941 | 0.830 | 0.904 | 0.925 | **0.9000** |
| descendente `[1, .5, .25, .125]` | 0.893 | 0.807 | 0.904 | 0.925 | 0.8823 |
| **ascendente `[1, 2, 4, 8]`** | 0.941 | **0.887** | **0.931** | **0.933** | **0.9230** |

602-633 parámetros. **+0.023 sobre el baseline** — más que cualquiera de las
nueve intervenciones de la sesión anterior, que no movían la media más de 0.02.

**RE-VERIFICADO tras el reinicio, corriendo de una en una** (las configs de
`/tmp` se habían perdido y se reconstruyeron desde el repo): las cuatro semillas
dan **el mismo valor y los mismos contadores**, así que no era un artefacto de
la ejecución en paralelo. Baseline s1 también reproduce exacto (0.830, fam=2).

| | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| acc | 0.941 | 0.887 | 0.931 | 0.933 |
| certs familia | 12 | 6 | 7 | 10 |
| epochs muertas | 7 | 10 | 3 | 2 |
| tiempo / RAM | 205 s | 172 s | 161 s | 178 s · 1.74 GB c/u |

12 min las cuatro en secuencial, contra ~9 min **por corrida** cuando competían
en paralelo.

**Descendente pierde por una razón estructural, no por ruido:** una η pequeña
certifica un paso pequeño, y un paso de familia certificado **retorna de
`grow_until_certified` en el acto**, así que compra un aplazamiento del
crecimiento sin progreso (lo que vigila `certify_family_min_gain`, aquí 0).
Además se autolimita: con η→0 el Δ realizado colapsa sobre el término de primer
orden, es decir sobre la proyección tangente, cuyo eps es ≥1/2 — que es por lo
que se llegó a la escalera. Medido en una llamada:
cos 0.7871 / 0.7955 / 0.7594 / 0.7481 para η 1 / 0.5 / 0.25 / 0.125.

**Ascendente pide más curvatura al clon y se paga.** Lo llamativo es lo poco que
hace falta: en las cuatro semillas **solo 4 pasos** certificaron por encima de
η=1 (η=2 tres veces, η=4 una, η=8 **nunca**). La ganancia **compone**: un rescate
temprano cambia toda la trayectoria y s1 pasa de 2 a 6 certificaciones de
familia, de las cuales solo una viene de η>1.

⚠️ **Aviso metodológico auto-infligido:** llamé refutado al ascendente mirando
las 12 primeras líneas del log de s1, donde los cosenos son planos (0.777 /
0.712 / 0.716 / 0.784). Sobre la corrida entera es el mejor resultado de la
sesión. **No concluir de logs parciales.**

## Lo que sí encontró la medición

**1. Las cuatro semillas se encallan en eps ≈ 0.501 y ahí se mueren.**

```
[CERTIFY] Epoch 15: growth target eps < 0.5 NOT reached (stopped at 0.5010
          after 0 growths, reason: parameter_budget); step certificate
          eps < 0.5 FAILS, so no rate is admissible
[FGD] Epoch 16 ... lr=0        (y las epochs 17-25 idénticas)
```

Gastado el presupuesto no puede crecer, así que eps no se mueve, así que no hay
tasa admisible, así que no pasa nada más. Las tres salidas fallan por un pelo:
eps 0.5010 contra 0.5, y el clon de la familia en cos 0.85 contra 0.866.

| tope 600 | encalla en | epochs muertas | acc |
|---|---|---|---|
| s0 | 0.5011 | 7 | 0.941 |
| s1 | 0.5010 | **11** | 0.830 |
| s2 | 0.5120 | 5 | 0.904 |
| s3 | 0.5040 | 4 | 0.925 |

Con presupuesto libre **ninguna encalla** (25 epochs completas): el tope es lo
que cierra la única salida abierta.

**2. s1 no subajusta por la arquitectura — es idéntica a la que sí gana.**

```
s3  14-19-13  601p  train_acc 0.991  -> test 0.925
s1  14-18-13  601p  train_acc 0.933  -> test 0.830
```

**3. La primera decisión de s1 es una ráfaga de 6, tomada en la epoch 2.**

```
eventos de crecimiento (presupuesto libre)      bloques  1er evento   acc
s0  e4:+1  e12:+4  e25:+1                          +6      e4 +1     0.949
s2  e5:+1  e7:+1  e11:+7  e20:+2 …                +16      e5 +1     0.945
s3  e5:+2  e15:+1  e19:+1  e25:+1                  +5      e5 +2     0.941
s1  e2:+6  e13:+4  e14:+3  e15:+4  e18:+3         +20      e2 +6     0.845
```

s1 es **la única** cuya primera decisión es una ráfaga, y la toma en `2-2-2`,
donde el sistema tangente tiene rango 25 y las medidas que eligen la forma son
ruido. Se compromete con una forma de inmediato y no se recupera: gasta más
parámetros que nadie (1142 contra los 815 de s3) y saca la peor accuracy. Con
presupuesto libre s1 interpola (`train_acc` 1.000, `train_loss` 0.0001) y aun
así testea 0.845 — ahí el problema ya no es subajuste sino generalización.

## La inicialización: medida, y REFUTADA

Arrancar en `5-5-5` y en `8-8-8` en vez de `2-2-2` (mismo tope 600, sin barrido):

| config (tope 600) | s0 | s1 | s2 | s3 | media | epochs muertas | certs familia |
|---|---|---|---|---|---|---|---|
| baseline `2-2-2` | 0.941 | 0.830 | 0.904 | 0.925 | **0.9000** | 7/11/5/4 | 12/2/5/8 |
| `5-5-5` | 0.898 | 0.848 | 0.912 | 0.917 | 0.8938 | 11/10/10/14 | 7/2/5/3 |
| `8-8-8` | 0.903 | 0.900 | 0.758 | 0.896 | 0.8642 | 16/18/14/15 | 3/1/0/1 |

**Arregla justo lo que predecía la hipótesis y pierde igual.** s1 sube
0.830 → 0.848 → **0.900** (sin ráfaga inicial sobre ruido, tal como se esperaba),
pero un arranque mayor **gasta presupuesto de entrada** — `8-8-8` son 193 de los
600 — choca contra el muro mucho antes, y las epochs muertas suben de 4-11 a
14-18 mientras las certificaciones de familia casi desaparecen. s2 se hunde a
0.758.

**Conclusión: `2-2-2` no era el problema. El problema es el muro de 600, y
arrancar mayor lo acerca.** El diagnóstico de la ráfaga (medido, correcto) y el
remedio (arrancar mayor) no son lo mismo.

## Lemma 3.5 sobre la distancia de la familia: colapsa sin `min_gain`

La familia **compromete el clon entero** — también con η=1, también en el
baseline —, así que su distancia nunca ha estado gobernada por el lema. Poner
`certify_family_lemma35_rate: true` con el barrido ascendente **colapsa**:

```
s1 0.163   s2 0.124   s3 0.170      (contra 0.887 / 0.931 / 0.933)
s1: 25 certificaciones de familia en 25 epochs, 0 crecimientos, sigue en 2-2-2
```

Escalado a la tasa del lema (~0.26) el paso casi no mueve `f`, el residuo casi
no cambia, la familia recertifica la misma dirección siempre y el crecimiento
no llega nunca. Es exactamente el fallo que documenta el comentario de
`certify_family_min_gain`. **Certificar no es progresar.**

Nota histórica: aquella variante de MNIST usaba `lemma35_rate: true` con
`certify_family_min_gain: 0.02`. La configuración canónica actual unificó ambos
campos con la base (`false` y `0.0`); no debe atribuirse aquel resultado al YAML
actual sin volver a medirlo.

## ¿Crece de forma no uniforme? NO — y el patrón es diagnosticable

Pregunta del usuario. Medido sobre las 4 corridas ascendentes.

**1. La forma final es NIVELADA, con tope y sin tope.** Ratio ancho/estrecho:

| | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| tope 600 | 15-16-15 (1.07) | 15-16-15 (1.07) | 15-17-15 (1.13) | 15-16-15 (1.07) |
| libre | 20-19-18 (1.11) | 21-22-23 (1.10) | 22-20-19 (1.16) | 16-22-15 (1.47) |

**2. La forma final no depende del arranque.** Las 4 semillas parten de
`[2,2,2]`, `[3,6,2]`, `[4,2,2]`, `[5,6,2]` y las 4 aterrizan en `15-16-15`.
Tres dan **602 params exactos** contra un tope de 600: **la forma la fija el
presupuesto, no los datos.**

**3. Las compras NO son round-robin — concentra y luego rellena.** s0:

```
e1:W2 e2:W2 e3:W2 e4:W2 | e6:J0 e7:W0 e8:J0 e9:J1 e10:W2 e11:W0 e12:W2
e13:W0 e14:J1 e15:W0 e16:W0 e17:J0 e18:W0 e19:W0
```

**4. La ráfaga gasta en la banda voluntaria.** s1 epoch 9: UN bloque en capa 2
baja eps 1.635 → 0.445, que **ya certifica**, y `certify_adaptive_growth` mete
**8 más** hasta 0.324. s2 epoch 11: certifica en 0.482 con uno, añade 7 más. La
parada no es "ya certifiqué" sino "sigo comprando mientras esta capa siga
siendo el mejor sitio por parámetro". En s0/s1/s2 la ráfaga deja la capa 2 en
**exactamente 15** y esa capa no vuelve a crecer nunca.

**5. NO es que el criterio no sepa decidir.** El ledger `[WHY]` da ganancias
distinguibles, típicamente 0.004-0.009 contra 0.0006 del peor candidato
(factor 10). Se decide por medición, no por empate ni ruido.

## LA CAUSA RAÍZ: eps es RELATIVO, así que nunca se entera de que terminó

Medido con `runs/cfg/datB_facil.yaml` — la función más fácil posible
(`active_features: 1`, `frequency: 0.5`, `interaction_strength: 0.0`),
presupuesto **libre**:

| epoch | forma | params | test_acc |
|---|---|---|---|
| 2 | (2,2,2) | 25 | 0.988 |
| 3 | (2,4,2) | 35 | 0.999 |
| **6** | **(4,5,4)** | **74** | **1.000** ← perfecta |
| 12 | (10,11,11) | 315 | 1.000 |
| **22** | **(15,16,15)** | **602** | 1.000 |
| 25 | (17,18,16) | 730 | 1.000 |

**Llega a accuracy perfecta con 74 parámetros y sigue creciendo hasta 730 — 10×
más — sin ganar nada.** 19 epochs comprando estructura con la tarea resuelta.

**Y en la epoch 22 pasa exactamente por `(15,16,15) = 602 params`**, que es
donde aterrizaban las 4 semillas con tope 600. Es decir: **`15-16-15` nunca fue
una elección.** Es el punto de una trayectoria monótona casi diagonal donde se
acabó el presupuesto. El tope no selecciona arquitectura: corta una cinta.

**Por qué — HAY DOS eps Y SE CONTRADICEN.** Misma fórmula
`‖g − r‖ / ‖g‖` (ojo: el denominador es la norma de la PROYECCIÓN, no la del
residuo — `tangent.py:1529-1540`), evaluada en dos puntos del espectro de
amortiguación:

| cantidad | dónde se mide | valor en `datB_facil` |
|---|---|---|
| eps del bucle `grow_until_certified` | amortiguación **mínima** del bracket | **0.15 – 0.39** → certifica |
| `rel_err` del paso | amortiguación **elegida** | **0.64 → 114** → falla siempre |

`growth_requires_admissibility_failure: true` dispara el crecimiento cuando
falla la admisibilidad **del paso**. Con amortiguación fuerte `g` recupera solo
una fracción del residuo y el cociente se dispara — **aunque `r` NO sea
pequeño**: en `datB_dificil`, con `train_loss 0.0134`, el `rel_err` llega a 229.
Así que el disparador lee "estructura desesperadamente inadecuada" mientras el
bucle de crecimiento, mirando la misma fórmula bien acondicionada, dice
**0.15 — sobra estructura**. Nadie escucha al segundo.

Correlación medida en `datB_facil`: **25/25 epochs con
`relative-error condition failed` y 25/25 con crecimiento.** El crecimiento no
correlaciona con ninguna medida de falta de capacidad.

### Confirmado en 6 corridas, presupuesto libre (`runs/cfg/dat{A,B}_*.yaml`)

| corrida | forma | params | ratio | acc | creció |
|---|---|---|---|---|---|
| B fácil (1 feat, f0.5, sin inter) | (18,19,20) | **872** | 1.11 | **1.000** | 25/25 |
| B difícil (4 feat, f3.0, inter 0.6) | (22,21,23) | 1123 | 1.10 | 0.313 | 17/25 |
| A muestra 0/2/1 | (20,19,18) | 878 | 1.11 | 0.949 | 24/25 |
| A muestra 11/12/13 | (21,21,21) | 1051 | 1.00 | 0.944 | 25/25 |
| A muestra 21/22/23 | (22,22,25) | 1217 | 1.14 | 0.910 | 25/25 |
| A muestra 31/32/33 | (20,21,23) | 1071 | 1.15 | 0.943 | 23/25 |

**Todas niveladas, todas 872-1217 params, todas creciendo 23-25/25.** La
diferencia entre tareas (872 vs 1123, 1.3×) queda ENTERRADA en la variación
entre muestras de la MISMA función (872 a 1217). **El método no dimensiona a la
tarea.**

Esto explica de golpe:

1. **El nivelado** — no elige formas niveladas; recorre una trayectoria fija y
   te quedas donde te pilla el tope.
2. **Los 602 params clavados en 3 de 4 semillas** — el tope corta la cinta
   siempre en el mismo punto.
3. **La ráfaga comprando 8 bloques tras certificar** — el mismo mecanismo.
4. **Por qué ayudó el barrido ascendente** — cada paso de familia certificado
   **aplaza un crecimiento**. No mejoraba la búsqueda de arquitectura: frenaba
   el gasto. Eso reencuadra `d843873`.

**DESCARTADAS por este experimento** (eran mis dos sospechosos): el operador
conjunto `growth_where_joint` y el divisor `growth_where_cost_exponent`. El
problema no está en CÓMO se rankean los candidatos sino en QUE SE SIGA
CRECIENDO.

**Dónde atacarlo** — hace falta un criterio de parada que sí sature. Candidatos,
sin medir todavía: (a) parar cuando el residuo absoluto `|r|` deje de bajar, no
su fracción; (b) exigir que el crecimiento pague en la sonda de validación, no
solo en eps de train; (c) `certify_growth_min_gain` sobre eps absoluto.
⚠️ Ninguna es gratis: el certificado `eps < min(threshold, 0.5)` es intocable,
así que el criterio de PARADA debe ir aparte, sin tocar el de ADMISIÓN del paso.

## Después, y falta entero

1. Validar que la arquitectura final es la de **mínimo número de parámetros**
   para su accuracy (comparar contra grid por forma, no contra `lr` fijo).
2. Validar que **no sesga hacia una arquitectura fija en otras bases de datos**
   — configs listas en `runs/cfg/mnist_base.yaml` y `runs/cfg/mnist_up.yaml`.
   **Ojo:** la variante histórica de MNIST usaba
   `certify_family_lemma35_rate: true` y `certify_family_min_gain: 0.02`; la
   configuración canónica actual ya no usa ese régimen.

## Advertencia metodológica

La dispersión entre semillas es de **11 puntos**; ninguna de las nueve
intervenciones probadas mueve la media más de 2. Con n=4 cualquier ranking de
reglas es provisional. Varias conclusiones de esta sesión se dieron por buenas y
luego resultaron artefactos de medición — **medir antes de afirmar**, y usar el
ledger `[WHY]` para distinguir "no gana" de "no se evaluó".
