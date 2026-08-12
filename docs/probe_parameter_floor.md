# La sonda dejó de crecer justo cuando P creció

Segundo defecto de la sonda de certificación, encontrado **después** de arreglar
el primero ([`probe_self_interpolation.md`](probe_self_interpolation.md)) y en
la corrida que ese arreglo hizo posible.

Todos los números vienen de `TAU-Frugal/stable-tiny`, corridas en `margpu008`.

---

## 1. El punto de partida: el primer arreglo funcionó

| brazo | run | epochs | test | tiempo |
|---|---|---|---|---|
| `probe-fixed` | `unguzxkq` | 8 | **0.7741** | 1.58 h |
| `probe-biased` | `euepsdrk` | 6 | 0.4337 | 2.31 h |

El techo de 0.350 que tenía la rama desapareció. Pero ninguno terminó.

## 2. Por qué murieron: dos causas, y ninguna era la que parecía

La telemetría de W&B lo zanja:

| | memoria GPU a lo largo del run | final |
|---|---|---|
| sesgado (`euepsdrk`) | 1.3 → 6.6 → 12.8 → **40.1 GB (93.4%)** | **OOM** |
| arreglado (`unguzxkq`) | 1.3 → 3.4 → 4.5 → 7.3 → **10.6 GB (24.7%)** | muere sin agotar nada |

El sesgado es un **OOM reproducible**: `yqfvy8l7`, `uesf9xgb` y `euepsdrk`
mueren en la MISMA epoch 6, con el mismo `acc 0.4337`, el mismo `P=9443` y las
mismas 2.31 h. Determinista, no desalojo — que es lo que se supuso al principio
por la coincidencia horaria con un array de conv.

El arreglado murió con la GPU al 24.7% y el RSS del host en 2.4 GB de 480: **ni
una ni otra se agotaron**, y no hay traza en el log. No se puede atribuir desde
la telemetría disponible, y por eso el lanzador ahora registra `sacct` en un
`trap EXIT`.

## 3. El defecto: el dimensionado de la sonda tiene un punto fijo

Independiente de la muerte, el certificado ya era falso desde ~la epoch 5.
MEDIDO en `unguzxkq`:

| P | NK | rank | **NK/P** | eps |
|---|---|---|---|---|
| 1612 | 3840 | 624 | 2.38 | 0.441 |
| 3383 | 5120 | 1249 | 1.51 | 0.459 |
| 4283 | 5120 | 1107 | 1.20 | 0.398 |
| 5992 | 5120 | 862 | **0.85** | 0.301 |
| 9383 | 5760 | 602 | **0.61** | **0.0131** |
| 12869 | 5760 | 161 | **0.45** | 0.0410 |

`eps` se desploma de 0.44 a 0.009 justo cuando `NK/P` cruza 1, y en esas mismas
líneas el `rel_err` medido en validación sigue en **1.02**.

**La causa es que el suelo se expresaba en `NK/rank` y no en `NK/P`.**
`_bounded_probe_batches` calcula `target_rows = kappa * rank`. Pero
`rank(J) ≤ NK` por construcción y el rango se **mide sobre la sonda actual**:
sonda pequeña → rango pequeño → «no necesitas más filas». Punto fijo
auto-confirmante, y la medición lo muestra cerrándose — el rango cae de 1249 a
161 mientras P triplica, así que el criterio pide **menos** filas justo cuando
hacen falta más.

Reproducido a través del propio dimensionador, sin suelo la petición no solo
deja de crecer, **colapsa**:

| P | rango medido | filas pedidas sin suelo | con suelo 1.25 |
|---|---|---|---|
| 1612 – 3383 | 624 – 1249 | 2560 – 5120 | **idéntico** |
| 4283 | 1107 | 4480 | 5760 |
| 9383 | 602 | 2560 | 12160 |
| 14487 | 132 | **640** | 18560 |

Solo el trinquete monótono sobre el tamaño de la sonda la mantuvo en 5760. Así
es como un certificado acabó leyéndose sobre 0.40 filas por parámetro.

El dimensionado por rango se introdujo para CIFAR, donde `rank << P` de verdad.
En MNIST completo elimina el único suelo que importaba — y el comentario del
config afirmaba que `kappa=4` «mantiene la invariante NK > P tras cada
crecimiento», cosa que la medición refuta.

## 4. Lo que se descartó al verificarlo

**`certify_stream_gram` es un no-op en esta ruta.** El plan proponía activarlo
para que la sonda pudiera crecer sin explotar la memoria. VERIFICADO
ejecutándolo: `exact_tangent_system` corta en su primera línea con
`if tuple(config.family_order) == ("matrix_free_tangent",)` y va a
`_matrix_free_tangent_system`, que nunca lee la bandera. Con ON y OFF sobre
`mnist_full.yaml`, `eps` es **bit-idéntico** (`0.9158695992197211`) y el sistema
sale factorizado `(960,216)/(1612,216)` en los dos casos.

Y de paso cae el argumento de memoria que lo motivaba: el sistema matrix-free es
**factorizado**, `(NK × r) + (P × r)`, no el denso `NK × P`. La memoria nunca fue
el muro, así que la rebaja del presupuesto de 100000 a 35000 que el plan
proponía queda retirada.

**El culpable real de los 40 GB es `matrix_free_block_chunk`**, y el comentario
del propio campo lo dice:

> *"El range finder pide un bloque de `rank + oversampling` direcciones — del
> orden de `P` — y `vmap` mantiene las activaciones completas de todas a la vez.
> MEDIDO ... el bloque pide decenas de GB y tumba la máquina."*

`mnist_full.yaml` lo tenía en **0 (sin acotar)**. La conv lo tiene en 32 y su
lanzador lo sube a 256 en A100. Encaja con la medición: el brazo sesgado acumuló
contraejemplos hasta NK=16000 contra 5760 del arreglado, 2.8x más muestras por
dirección, y es el único de los dos que reventó.

Trocear **no puede cambiar el resultado** — las direcciones del bloque son
independientes, así que es un bucle; en coma flotante se mueve un ulp.

## 5. El cambio

| campo | qué hace | default |
|---|---|---|
| `certify_probe_parameter_floor` | `NK ≥ floor · P`, acotado por el dataset | `0.0` |
| `matrix_free_block_chunk` | acota el bloque del range finder | `0` |

Más el diagnóstico: `[CERTIFY-PROBE]` imprime `NK/P` y grita cuando `NK < P`,
con el contador `probe_below_parameter_floor`. Tardó dos corridas completas en
verse porque había que dividir dos números a mano. Lee las **filas reales de la
sonda**, no `system.target`, que bajo un sistema sustituto sería otra cantidad.

### Por qué no daña lo que iba bien

Era la restricción explícita, y se cumple por construcción: **el suelo es
inactivo mientras la sonda ya lo cumple**. Verificado replicando el
dimensionador sobre la trayectoria real — de P=1612 a P=3383, donde `NK/P` iba
de 2.38 a 1.51, el resultado es **idéntico**, y la primera intervención cae en
P=4283, justo donde el run empezaba a degradarse.

## 6. Qué mira la corrida

`cluster/slurm/mnist_full_probe.sbatch`, un solo brazo. Criterio, en orden:

1. **`NK/P ≥ 1.25` durante toda la corrida.** Hoy cae a 0.45.
   ```bash
   grep -oE "NK/P=[0-9.]+" slurm_logs/<log> \
     | sed 's/NK\/P=//' | sort -n | head -1
   ```
   Y `grep -c "BELOW INTERPOLATION FLOOR"` tiene que dar **0**.
2. **`eps` no se desploma** por debajo de 0.01 mientras el `rel_err` de
   validación siga por encima de 1. Hoy pasa: eps 0.0131 contra rel_err 1.02.
3. **La corrida termina las 40 epochs** en vez de morir a mitad.
4. **`test_acc` supera claramente 0.774**, que es donde estaba al morir.

## 7. Sobre 0.98

No se promete. Lo que se puede decir con lo medido: el presupuesto de 100000
parámetros sigue en pie —la memoria no era el muro— y el límite pasa a ser el
reloj. `exact_tangent_system` se lleva ~90% del tiempo (547 s de ~605 s en el
perfil conv, 7.60 s por construcción contra 0.94 s en el lineal), así que si la
corrida termina las 40 epochs y se queda corta, el frente siguiente es el coste,
no otro parche a la sonda.
