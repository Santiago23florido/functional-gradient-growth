# La sonda fija se auto-interpola

Diagnóstico medido del estancamiento de MNIST lineal matrix-free
(`configs/experiments/mnist_full.yaml`), y de los tres arreglos que salen de él.

Todos los números vienen de corridas en `margpu008` publicadas en
`TAU-Frugal/stable-tiny`. Los identificadores son los de W&B.

---

## 1. El síntoma

`mnist-full-guard-off-seed-0` (`2kbo8rf4`), 40 epochs:

```
ep1  train_loss=0.3580  test=0.090
ep3  train_loss=0.2781  test=0.356
ep5  train_loss=0.2639  test=0.350
ep6  train_loss=0.2639  test=0.350
...                                  <- 35 epochs
ep40 train_loss=0.2639  test=0.350
```

No es convergencia lenta. Es el **mismo valor dígito a dígito** durante 35
epochs: cero pasos comprometidos, cero crecimientos, un no-op exacto.

## 2. El crecimiento NO es patológico

Era la primera sospecha, y la distribución de crecimientos la alimenta:

| run | L0 | L1 | L2 | arquitectura final | test |
|---|---|---|---|---|---|
| `2kbo8rf4` sonda fija | **0** | 6 | 16 | `784->6->8->18->10` | 0.350 |
| `yqfvy8l7` con refine | **9** | 17 | 17 | `784->11->19->19->10` | 0.434 |

El cuello de botella de expresividad medido en la epoch 1 dice
`L0=9.39e-01`, `L1=4.15e-04`, `L2=3.25e-03` — tres órdenes de magnitud a favor
de L0, la capa que recibe las 784 entradas — y aun así la sonda fija metió
**cero** neuronas ahí y 16 en L2. Un `784->6` no puede representar MNIST, y ese
es exactamente el techo de 0.35.

Pero el criterio no está roto. `_select_growth_candidate`
(`src/fgdlib/search/certify.py`) es el argmin exacto de eps sobre todas las
localizaciones, que es **el criterio del teorema**. Lo que estaba mal es el eps
que lee.

**La prueba:** al reparar la medición el WHERE se recolocó solo, sin tocar el
criterio. L0 pasó de 0 a 9 crecimientos y la primera capa de 6 a 11 neuronas.
No hace falta un criterio nuevo ni una patología que evitar.

## 3. La causa raíz

La base de la sonda son 704 imágenes de 50000 (1.4%), fijadas una vez para toda
la corrida. Los pasos certificados se **ajustan sobre ella**, así que la
interpolan y deja de representar a la población.

Funcional por imagen, sonda contra train completo, a lo largo de la corrida:

| transacción | `2kbo8rf4` sonda fija | `yqfvy8l7` con refine |
|---|---|---|
| 1 | 1.0x | 1.0x |
| 25 | 3.6x | 1.2x |
| 50 | **14.1x** | 1.2x |
| 100+ | **14.2x, congelado** | 1.1–1.3x |

En régimen la sonda está en `0.1859`/imagen y la población en `2.6388`/imagen.
La sonda está prácticamente resuelta y el 99.1% de los datos intacto.

**El repo ya tenía escrito este diagnóstico** en `src/stable_tiny/pipeline.py`,
en el comentario de `probe_resample`:

> *"A FIXED probe turns the method into Newton's method on one subsample: the
> residual on those samples is driven to zero and the network interpolates them.
> [...] RelErr is normalised by ‖g‖, so it diverges precisely when there is no
> residual left on the probe to approximate. Drawing a fresh probe each outer
> step makes the functional gradient an unbiased estimate of the one over the
> dataset, which is the object the theory is about."*

Coincide punto por punto, incluido el `rel_err` de epoch entre 1.02 y 1.55
durante todo el run. Y la config lo desactivaba con una justificación **falsa a
esta escala**, heredada literal de `configs/fgd/mnist_matrix_free.yaml` (donde
la cobertura era del 7%): *"probe_batches covers the entire training set"*.

Peor: `validate_probe_refinement` **exigía** `probe_resample: false`, así que
activar el refinamiento forzaba justo el régimen contra el que advierte el
comentario.

### Los tres síntomas de esa única causa

**(a) El certificado miente.** `eps=0.288` sobre la sonda contra `rel_err=1.02`
medido en validación. Toda transacción baja la sonda y sube el train completo:

```
[TRANSACTION] functional=8.33e1->7.95e1  full_train=1.3194e5->1.3274e5  accepted=False
```

las cuatro retries, siempre.

**(b) El WHERE se desvía.** El argmin de eps se calcula sobre esa sonda
interpolada, donde ampliar la última capa oculta baja el residuo de mínimos
cuadrados más barato que ampliar la de 784 entradas. De ahí el `784->6`.

**(c) El paro es un punto fijo.** En la epoch 20 se imprime, idéntico, 35 veces:

```
[FGD] Epoch 20 ... lr=0, rel_err=0.356
[CERTIFY-PROBE] P=5143 NK=4480 rank=528 NK/rank=8.4848 eps=0.288351
[REALIZABLE-GROWTH] current_eta=none current_certified_progress=0.000000e+00
[REALIZABLE-GROWTH] location=0 candidate_eta=none candidate_certified_progress=0.000000e+00 delta_progress=-0.000000e+00
[TRANSACTION] retry=0..3  accepted=False
[LINEAR] eta 0.1107 -> none (defect 5.148e-01, 14 backtracks)
```

Las tres puertas cerradas a la vez:

- `eps=0.288 < 0.5` → el certificado se cumple → `while epsilon >= target` no corre.
- `rel_err=0.356 < 0.5` → la admisibilidad se cumple → el crecimiento unificado de epoch no dispara (`growth_requires_admissibility_failure: true`).
- `certify_realizable_progress_growth` es la única ruta viva, y se auto-bloquea.

Ese último es un defecto propio: `_select_realizable_progress_growth` exige que
el **clon crecido** realice progreso finito. Cuando el modelo base no puede
realizar ningún paso, el clon de +1 neurona tampoco puede, así que `finite` es
False, `delta_progress` sale como un `-0.0` con signo que solo *parece* un
veredicto, se rechaza, y `grow_until_certified` **retorna** en vez de ceder el
turno.

Es exactamente la invariante que la ruta ordinaria ya afirma en voz alta doce
líneas más abajo:

> *"growth refused while no step can certify"* — *"Loud rather than frozen:
> refusing here while no step certifies would be the deadlock this whole design
> exists to avoid."*

## 4. Por qué NO reinicialización de pesos

Se planteó y se descarta con la medición. Los pesos no están en una cuenca mala:
sobre lo que se le mostró al modelo el funcional está en `0.186`/imagen,
prácticamente resuelto. Lo que está mal es **que se le mostró el 0.9% de los
datos**. Reinicializar tiraría una función que es correcta sobre su muestra,
rompería la preservación de función de la que dependen todos los certificados, y
volvería a caer en el mismo pozo en ~50 transacciones, que es lo que tarda la
razón en pasar de 1.0x a 14.1x. El arreglo va en la medición.

## 5. Los tres arreglos

Ninguno introduce un criterio nuevo: dos restauran algo que el repo ya argumenta
y el tercero restaura una invariante que el repo ya afirma.

| campo | qué hace | default |
|---|---|---|
| `certify_probe_base_resample` | redibuja la BASE cada outer step y conserva los contraejemplos | `false` |
| `certify_probe_refine_max_rows` | acota la memoria de contraejemplos por filas con desalojo del menos violador, en vez del tope terminal de rondas | `0` |
| (sin campo) | el criterio realizable **se abstiene** cuando ambos extremos puntúan cero, y cae al argmin de eps | — |

La monotonía que el refinamiento necesita es sobre los **contraejemplos**, no
sobre la base; el conjunto de huellas se reconstruye como la unión para que el
dedup siga cubriendo las dos mitades. `train_loader` es `shuffle=True`, así que
leer sus primeros lotes ya da una muestra fresca: no hace falta muestreador
nuevo.

La abstención no puede desencadenar crecimiento infinito: el estado solo se
alcanza cuando **ningún** eta realiza un paso, y en cuanto uno lo hace
`current_certified_progress` es positivo y vuelve a mandar la comparación
estricta. El tope de parámetros sigue acotando por encima.

## 6. Qué mira el A/B

`cluster/slurm/mnist_full_probe_ab.sbatch`, brazo 0 arreglado contra brazo 1 con
la sonda de hoy. Los dos llevan la abstención, que no es opcional: sin ella el
brazo sesgado se congela 35 epochs y no mide nada.

El instrumento nuevo es la razón sonda/población:

```bash
awk '/CERTIFY-PROBE/{ if (match($0,/NK=[0-9]+/)) nk=substr($0,RSTART+3,RLENGTH-3)+0 }
     /TRANSACTION/{ if (match($0,/functional=[0-9.e+]+/)) f=substr($0,RSTART+11,RLENGTH-11)+0
                    if (match($0,/full_train=[0-9.e+]+/)) ft=substr($0,RSTART+11,RLENGTH-11)+0
                    if (nk>0) printf "%.1fx\n", (ft/50000)/(f/(nk/10)) }' slurm_logs/<log>
```

**Pasa si** las cuatro a la vez:

1. La razón se mantiene **≤ 1.5x** toda la corrida (hoy: 14.2x desde la transacción ~50).
2. `train_loss` cambia en cada epoch — cero epochs repetidas dígito a dígito (hoy: 35 seguidas en 0.2639).
3. `L0` recibe crecimientos y la arquitectura final no es un `784->6` (hoy: 0 en L0).
4. `test_acc` monótona y claramente por encima de 0.35.

**Refuta si** la razón se queda baja y aun así la accuracy se estanca: entonces
la sonda no era la causa raíz y hay que volver a medir antes de tocar nada más.

### Nota sobre reproducirlo en pequeño

No se puede. El sesgo necesita que la sonda sea una fracción pequeña de los
datos: con `mnist_train_samples: 640` la sonda cubre ~60% del train y la razón
se queda en 1.0x en los dos brazos, medido localmente. Es coherente con el
diagnóstico y es la razón de que la medición decisiva tenga que ser a escala
completa.

## 7. Lo que no se toca

- `eps < min(rel_error_threshold, 0.5)` estricto; no se relaja ninguna condición.
- Lema 3.5 y su intervalo de tasa; `mu`, `L_s`, la Prop. 3.8.
- Preservación de función en todo crecimiento.
- El criterio WHERE: sigue siendo el argmin exacto de eps. Se arregla lo que mide, no cómo decide.
- `configs/fgd/*.yaml` queda bit a bit. Los campos nuevos nacen desactivados y
  el validador los restringe a `family_order: [matrix_free_tangent]` con el
  refinamiento activo. Regresión N1024 verificada exacta:
  `acc 0.9352 / 529 params / widths (10,18,14) / 23 crecimientos`.
