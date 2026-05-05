# PMDKernel

Kernel CUDA en Julia para calcular el campo magnético $B_0$ de imanes permanentes
modelados como dipolos. Aplicación principal: simulación del **OSI² OpenMRI**
(Open Source Imaging Initiative).

El repo contiene tres capas:

1. **Kernel CUDA + simulación** (`B0.jl`, `kernel.jl`) — Biot-Savart sobre dipolos
   en cualquier conjunto de puntos.
2. **Pipeline de datasets** (`generate_dataset.jl`) — perturbar geometría + correr
   kernel + escribir HDF5 con campo en grilla 3D y en sensores. Fuente de datos
   para entrenar redes.
3. **Modelos de ML en Python** (`Python/Models/<NOMBRE>/`) — paquetes con
   `data/model/train/metrics`. Notebooks homónimos en `Python/` los consumen
   como wrappers visuales para iterar.

---

## Convención de dimensiones

Todos los archivos del pipeline (Julia y Python) usan la misma notación. Esta
sección es la fuente de verdad — cualquier iteración nueva debe respetarla.

### Pipeline / dataset / red

| Símbolo | Significado                                       | Índice          |
|---------|---------------------------------------------------|-----------------|
| `N`     | nº de muestras (datasets)                         | n ∈ {1..N}      |
| `I`     | nº de sensores                                    | i ∈ {1..I}      |
| `J`     | nº de puntos de grilla (= Nx·Ny·Nz aplanado)      | j ∈ {1..J}      |
| `Nx,Ny,Nz` | nº de puntos por eje (grilla cartesiana)       | —               |
| `Nr,Nθ,Nz` | nº de puntos por eje (grilla cilíndrica)       | —               |

El mapeo que aprende la red v1 es **`B_sens (N, I, 3)` → `B_grid (N, J, 3)`**:
de las mediciones en los `I` sensores, predecir el campo en los `J` puntos de la
grilla 3D. Aplanado a la red: `(N, I·3) → (N, J·3)`.

### Kernel CUDA (genérico, no distingue sensores de grilla)

| Símbolo | Significado                  |
|---------|------------------------------|
| `n`     | puntos de evaluación (cualquier conjunto: `n = I` para sensores, `n = J` para grilla) |
| `m`     | dipolos / imanes             |

El kernel toma `(R[n,3], P[3,m], M[3,m])` y devuelve `B[n,3]`. No sabe si
estás evaluando en sensores o en grilla — eso lo decide el caller.

### Ejemplos de shapes

| Tensor          | Shape         | Unidades |
|-----------------|---------------|----------|
| `B_grid` (HDF5) | `(N, J, 3)`   | mT       |
| `B_sens` (HDF5) | `(N, I, 3)`   | mT       |
| Vista 3D grilla | `(N, Nx, Ny, Nz, 3)` | mT |
| Entrada FCNN    | `(N, I·3)`    | mT       |
| Salida  FCNN    | `(N, J·3)`    | mT       |

---

## Estructura del repo

```
PMDKernel/
├── B0.jl                  Pure B0(R,P,M) + helpers (cargar_imanes, construir_R)
├── kernel.jl              CUDA kernel _Bnu! (Biot-Savart, 4× unrolled)
├── generate_dataset.jl    Pipeline: perturb → B0(grid) + B0(sens) → HDF5
├── run_v1_dataset.jl      Driver de ejemplo (config + call a generate_dataset)
├── utils/
│   ├── calibracion.jl     Constantes nominales (NOMINAL_SCALE_1/2)
│   ├── perturb.jl         perturb(; kind, sigma_deg, ...) → (P, M) en memoria
│   ├── sensores.jl        R_sensores() → posiciones xyz mm de los I sensores
│   ├── timer.jl           StepTimer para benchmarks por paso
│   ├── gru.jl             Slicer3D + mostrar_grilla (visualización)
│   ├── bench.jl           Benchmark sintético del kernel
│   └── disco_DEPRECATED.jl  Pipeline viejo (no extender)
├── data/
│   ├── B0.npz                       Geometría nominal de imanes
│   ├── coordenadas_sensores.csv     Coordenadas (r_cm, θ°, z_cm) de los I sensores
│   ├── datasets/                    Salida HDF5 de generate_dataset
│   └── modelos/                     Checkpoints PyTorch (.pt)
└── Python/
    ├── v1_fcnn.ipynb                Wrapper visual de Models/v1_fcnn
    └── Models/
        └── v1_fcnn/                 Paquete: data, model, train, metrics
```

---

## Pipeline de generación de datasets (Julia)

`generate_dataset.jl` es el entry point. Para cada muestra `n = 1..N` corre:

```
perturb(seed = seed_base + n)  →  P, M
B0(R_grid, P, M)               →  B_grid    (J, 3)  Tesla
B0(R_sens, P, M)               →  B_sens    (I, 3)  Tesla
HDF5.write                     →  fila n del archivo
```

`R_grid` y `R_sens` se calculan una sola vez al inicio (no cambian entre muestras).
Las salidas se almacenan en mT.

### Uso básico

```julia
include("generate_dataset.jl")

config = (
    name      = "mi_experimento",
    n_samples = 500,
    seed_base = 0,
    perturb   = (kind=:both, sigma_deg=1f0,
                 mu1=2.035f0, sigma1=0.1f0,
                 mu2=8.48f0,  sigma2=0.85f0),
    grid      = (coords=:cartesian, x=-100:10:100, y=-100:10:100, z=-225:10:225),
    out_dir   = joinpath(@__DIR__, "data", "datasets"),
)

generate_dataset(config; verbose=true)
```

Resultado: `data/datasets/mi_experimento.h5`.

### Estructura del HDF5

| Dataset            | Shape          | dtype   | Unidades | Descripción                              |
|--------------------|----------------|---------|----------|------------------------------------------|
| `B_grid`           | `(N, J, 3)`    | float32 | mT       | Campo en cada punto de la grilla         |
| `B_sens`           | `(N, I, 3)`    | float32 | mT       | Campo en cada sensor                     |
| `R_grid_xyz_mm`    | `(J, 3)`       | float32 | mm       | Posiciones xyz de los puntos de grilla   |
| `sens_xyz_mm`      | `(I, 3)`       | float32 | mm       | Posiciones xyz de los sensores           |
| `grid_x/y/z`       | `(Nx,)` etc.   | float32 | mm       | Ejes del meshgrid (uno por dimensión)    |

**Layout de aplanamiento de la grilla:** orden x-outer, z-inner (loop Julia
`for xi in x, yi in y, zi in z`). En NumPy:
`B_grid.reshape(N, Nx, Ny, Nz, 3)` con C-order. Verificable contra
`np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")` (lo hace el loader del
notebook).

**Atributos del archivo:** `kind`, `sigma_deg`, `mu1/2`, `sigma1/2`, `n_samples`,
`seed_base`, `grid_coords`, más resumen de tiempos (`<paso>_mean_ms`, `_total_s`,
etc.) generado por `StepTimer`.

### Variantes

```julia
# Solo benchmark, sin escribir HDF5 (estima cuánto tarda el experimento real):
benchmark_dataset(config; warmup=2)

# Particionar en múltiples archivos (resumible si se cae):
generate_dataset_chunked(config; n_total=10_000, chunk_size=1_000)
# → mi_experimento_part01.h5 .. _part10.h5
```

### Grillas no cartesianas

```julia
grid = (coords=:cylindrical, r=0:5:160, theta=0:10:350, z=-60:5:60)
```

---

## Modelos en Python

Cada iteración de modelo vive en su propio paquete bajo `Python/Models/<NOMBRE>/`,
con la siguiente estructura mínima:

```
Models/<NOMBRE>/
├── __init__.py
├── data.py     # load_dataset(h5_path), split_and_normalize(...), make_loader(...)
├── model.py    # clase del modelo (subclass de nn.Module) + build_model()
├── train.py    # train(model, loader_tr, loader_va, ...) → dict(history, best_state, best_val)
└── metrics.py  # predict(...), metrics(y_true, y_pred), report(name, m)
```

El **notebook homónimo** (`Python/<NOMBRE>.ipynb`) sirve sólo para iterar
visualmente: importa los módulos y orquesta prints/plots. **No define lógica
entrenable.** Esto permite ejecutar lo mismo en headless sin Jupyter (cluster
HPC).

### Crear una iteración nueva

1. `mkdir Python/Models/v2_<algo>`, copiar `__init__.py` y los cuatro módulos
   desde `v1_fcnn` como base.
2. Ajustar `model.py` (arquitectura), `data.py` si cambia el preprocesamiento, y
   `train.py` si cambia el optimizador / scheduler / loss.
3. Crear `Python/v2_<algo>.ipynb` que importe del nuevo paquete.
4. Componentes compartidos entre iteraciones → refactorear a
   `Python/Models/_common/` antes de duplicar.

### Iteración actual

`v2_pinn` — Función de campo condicional `f(B_sens, x, y, z) → B(x,y,z)`. FCNN
de capas crecientes (~280k params) con activación SiLU. Loss combinada:

- **Datos**: MSE vs `B_grid` del simulador.
- **Maxwell**: `∇·B = 0` y `∇×B = 0`, vía `torch.autograd.grad` sobre las
  coordenadas (regiones libres de corrientes).
- **TV**: `mean(|∇B|)` para suavizado espacial.

Cada step de SGD ve **una sola configuración** con K puntos random — necesario
para que las losses físicas tengan sentido sobre un campo coherente.

Iteración anterior (`v1_fcnn`) abandonada: era un mapeo `B_sens (540) →
B_grid_aplanada (60.858)` puramente data-driven (MSE solo), con ~63M params.
La arquitectura de v2 es ~225× más chica y aprende un campo continuo.

---

## Cómputo: local vs cluster

- **Local (PC del usuario, Windows):** smoke tests del pipeline con datasets
  chicos (N=500–5k). Iteración rápida en notebook.
- **Cluster HPC universitario (Linux, headless, SSH):** entrenamiento real con
  N=100k+. Sólo scripts, no notebooks.

El código se escribe pensando en escalar desde el día 1: paths cross-OS, lazy
loading cuando aplique, y cualquier `train.py` debería ser ejecutable también
como `python -m Models.v<…>.train --config …` (a definir cuando llegue el
momento).

---

## Layout de arrays en el kernel

| Array | Shape   | Layout       | Unidades | Rol                    |
|-------|---------|--------------|----------|------------------------|
| `R`   | `[n,3]` | row-major    | mm       | Puntos de evaluación   |
| `P`   | `[3,m]` | column-major | mm       | Posiciones de imanes   |
| `M`   | `[3,m]` | column-major | A·m²     | Momentos magnéticos    |
| `B`   | `[n,3]` | row-major    | T        | Campo de salida        |

La asimetría row/column-major es intencional: garantiza accesos coalescentes en
GPU. El kernel usa unrolling de 4 dipolos por iteración (sin shared-memory
tiling). `B0.jl` convierte mm → m internamente — la API de usuario es siempre mm.

`R` puede ser cualquier conjunto de puntos: `R = R_grid` (para evaluar el
volumen) o `R = R_sens` (para evaluar sólo en los sensores). El kernel no
distingue.

### Singularidad y clamp

Las contribuciones cerca de los dipolos pueden divergir (`1/r³`); el kernel
clampea la magnitud de salida a `B_MAX_T = 0.06 T` (60 mT) preservando dirección.
La guarda interna es `r² > 1e-18`.

---

## Convención del eje X

El proyecto usa `+x` creciendo de **derecha a izquierda** en el resonador
(convención del hardware del OSI² OpenMRI). En coordenadas cilíndricas:
`x = -r·sin(θ)`, `y = r·cos(θ)`. Plots XY deben invertir el eje X (`ax.invert_xaxis()`)
para respetar la orientación física.

---

## Calibración

`utils/calibracion.jl` define las constantes nominales:

| Constante           | Valor   | Aplica a                |
|---------------------|---------|-------------------------|
| `NOMINAL_SCALE_1`   | 2.035f0 | array2 (1936 imanes)    |
| `NOMINAL_SCALE_2`   | 8.48f0  | array4 (384 imanes)     |

`cargar_imanes(; raw=true)` (en `B0.jl`) aplica estas escalas al leer el NPZ.
`perturb` recibe medias `mu1/mu2` que típicamente coinciden con estas constantes
(perturbación con baseline calibrado) o las sustituye (estudio de sensibilidad).
**No duplicar estas constantes en otros archivos** — importar siempre desde
`utils/calibracion.jl`.
