# PMDKernel

This repository contains a custom CUDA Julia kernel optimized for calculating the magnetic field of large sets of permanent magnets, modeled as dipole moment vectors, in any point in space.

As an application, `B0.jl` provides a full simulation of the magnetic field $B_0$ of the Open Source Imaging Initiative's (OSI²) OpenMRI.

---

## Estructura

| Archivo                    | Rol                                                                 |
|----------------------------|---------------------------------------------------------------------|
| `B0.jl`                    | Entry point: grilla 3D completa del resonador + (opcional) sensores |
| `kernel.jl`                | Kernel CUDA (Biot-Savart, tiling en memoria compartida)             |
| `utils/sensores.jl`        | Evalúa el campo en todos los sensores del CSV                       |
| `utils/disco.jl`           | `simular_disco` (pura) + `simular_disco_batch` (N muestras → NPZ)   |
| `utils/perturb.jl`         | `perturb` (rotación / magnitud / both) + `perturb_batch` (N seeds)  |
| `utils/bench.jl`           | Benchmark del kernel                                                |
| `utils/gru.jl`             | Visualización 3D (slicer ortogonal)                                 |
| `data/B0.npz`              | Geometría nominal de imanes                                         |
| `data/coordenadas_sensores.csv` | Coordenadas cilíndricas de sensores (radio cm, θ°, profundidad cm) |
| `data/simulaciones/`       | Carpeta de salida para todos los NPZ generados                      |

---

## Uso básico

```julia
include("B0.jl")

B0(true)                  # Grilla 121³ a 1 mm + visualización 3D
B0(true; sens=true)       # + campo en los sensores
B0(false)                 # Benchmark sin GUI
```

Salidas:
- `data/simulaciones/B_field.npz` — campo en la grilla completa
- `data/simulaciones/Bsensores.npz` — campo en todos los sensores

---

## Generación de datasets para ML (discos)

Para entrenar un modelo de ML sin tener que simular el volumen completo en una
tanda, dividimos el resonador en **discos**: losas delgadas (1 mm de grosor)
perpendiculares al eje Z. Cada disco produce un par:

- **entrada** → campo B en una grilla X-Y dentro del resonador a la Z elegida
- **salida**  → campo B en los sensores cuya Z cae dentro del disco

El pipeline tiene dos pasos: `perturb` genera NPZ de imanes perturbados y
`simular_disco` corre el kernel sobre ellos. Los NPZ guardan los momentos **ya
escalados** por los factores nominales; `simular_disco` los lee tal cual.

### Paso a paso

**1. Abrir la REPL de Julia en la raíz del repo** 

```bash
julia --project=.
```

**2. Definir las constantes requeridas por el kernel** (`BATCH_M` = dipolos por
tile de shared memory, `T` = dtype). Deben existir en el scope *antes* de
incluir `kernel.jl`:

```julia
const BATCH_M = 64
const T       = Float32
```

**3. Incluir el kernel y las utilidades** del pipeline de discos:

```julia
include("PMDKernel/kernel.jl")
include("PMDKernel/utils/perturb.jl")
include("PMDKernel/utils/disco.jl")
```

**4. Generar N sets de imanes perturbados** (seeds `1..N` por default). Cada
set se guarda como NPZ en `PMDKernel/data/perturbed/`:

```julia
paths = perturb_batch(5000; kind=:both, sigma_deg=1f0)
```

**5. Correr el kernel sobre los N sets** para el disco en la Z elegida. Se
apilan las muestras y se escribe un único NPZ consolidado:

```julia
simular_disco_batch(0f0, paths;
    out_path="PMDKernel/data/simulaciones/disco_z0_both_n5000.npz")
```

> `simular_disco` (sin `_batch`) es pura: corre una sola muestra y devuelve un
> `NamedTuple` con los arrays sin escribir nada.

### Capas Z de los sensores

Con el CSV actual (`coordenadas_sensores.csv`), los sensores se agrupan en
5 profundidades (cm), convertidas a Z (mm) via `(depth − 22) × 10`:

| depth (cm) | z (mm) | Nº sensores por capa |
|------------|--------|----------------------|
| 9.2        | −128   | 36                   |
| 15.6       | −64    | 36                   |
| 22         |   0    | 36                   |
| 28.4       |  +64   | 36                   |
| 34.8       | +128   | 36                   |

### Estructura del NPZ consolidado

Formato mínimo: solo los tensores que consume la red (`B_grid`, `B_sens`) y los
metadatos cuya reconstrucción en Python requeriría releer `coordenadas_sensores.csv`
(`sens_xyz`, `z_disco`).

| Clave      | Forma           | dtype   | Unidades | Descripción                                  |
|------------|-----------------|---------|----------|----------------------------------------------|
| `B_grid`   | (N, Nx, Ny, 3)  | float32 | mT       | Campo en la grilla XY (canales Bx, By, Bz)   |
| `B_sens`   | (N, K, 3)       | float32 | mT       | Campo en los K sensores del plano            |
| `sens_xyz` | (K, 3)          | float32 | mm       | Posiciones cartesianas de los sensores       |
| `z_disco`  | escalar         | float32 | mm       | Plano z del disco                            |

La grilla XY se reconstruye en el consumidor con `xy_step` y `grid_range` (el
notebook expone `XY_STEP`, `XY_MIN`, `XY_MAX` en su celda de config y valida vía
`assert` que la shape resultante coincide con `B_grid`). No se persisten `grid_xy`,
`sens_idx` ni `seeds`: los dos primeros son reconstruibles sin CSV y el último
queda implícito en los filenames de los NPZ de imanes (`B0_*_seed{N}.npz`).

Con `xy_step=10` y `grid_range=(-150, 150)` → `Nx=Ny=31` (961 puntos). Con el
CSV actual, cada capa contiene `K = 36` sensores.

### Parámetros

**`perturb` / `perturb_batch`**

| Parámetro     | Default          | Descripción                                            |
|---------------|------------------|--------------------------------------------------------|
| `n`           | requerido        | Nº de muestras (solo `perturb_batch`)                  |
| `kind`        | `:both`          | `:rotation`, `:magnitude` o `:both`                    |
| `sigma_deg`   | `1f0`            | σ angular (grados) de la rotación XY                   |
| `mu1,sigma1`  | `2.035 ± 0.1`    | Media/σ de la magnitud del array2 (1936 imanes)        |
| `mu2,sigma2`  | `3.051 ± 0.3`    | Media/σ de la magnitud del array4 (384 imanes)         |
| `seed`        | aleatorio        | Semilla RNG (`perturb`)                                |
| `seed_base`   | `0`              | Seeds usadas: `seed_base+1 … seed_base+n` (`_batch`)   |
| `out_path`    | requerido        | Ruta del NPZ (solo `perturb`)                          |
| `out_dir`     | `data/perturbed` | Carpeta de salida (solo `perturb_batch`)               |
| `name_prefix` | `"B0"`           | Prefijo del filename: `<prefix>_<kind>_seed<i>.npz`    |

**`simular_disco` / `simular_disco_batch`**

| Parámetro    | Default       | Descripción                                        |
|--------------|---------------|----------------------------------------------------|
| `z_mm`       | requerido     | Z del disco (mm)                                   |
| `xy_step`    | `10`          | Paso de la grilla en X e Y (mm)                    |
| `grid_range` | `(-150, 150)` | Rango X-Y de la grilla (mm)                        |
| `thickness`  | `1`           | Grosor del disco en Z (mm)                         |
| `threads`    | `256`         | Hilos CUDA por bloque                              |
| `data_path`  | requerido     | NPZ de imanes (solo `simular_disco`)               |
| `out_path`   | requerido     | Ruta del NPZ consolidado (solo `_batch`)           |

---

## Layout de arrays en el kernel

| Array | Shape  | Layout       | Unidades | Rol                    |
|-------|--------|--------------|----------|------------------------|
| R     | [n, 3] | row-major    | m        | Puntos de evaluación   |
| P     | [3, m] | column-major | m        | Posiciones de imanes   |
| M     | [3, m] | column-major | A·m²     | Momentos magnéticos    |
| B     | [n, 3] | row-major    | T        | Campo de salida        |

La asimetría row/column-major es intencional: garantiza accesos coalescentes en
GPU y tiling eficiente en memoria compartida (`BATCH_M = 64` dipolos por tile).

## Requisitos de scope

`BATCH_M` y `T` deben estar definidos en el scope **antes** de incluir
`kernel.jl` o cualquier utilidad. `B0.jl` los define al incluirse:

```julia
const BATCH_M = 64
const T       = Float32
```
