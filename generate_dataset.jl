using HDF5
using CUDA
using Printf
include("B0.jl")
include("utils/sensores.jl")
include("utils/perturb.jl")
include("utils/timer.jl")

"""
    generate_dataset(config; verbose=false) -> String

Pipeline live: para cada muestra `i = 1..N`,
    perturb(seed = seed_base + i) → B0(R_grid) + B0(R_sens) → escribir HDF5.

`config::NamedTuple` con campos:
- `name::String`              — usado en el path de salida
- `n_samples::Int`
- `seed_base::Int`            — semillas: seed_base+1 .. seed_base+n_samples
- `perturb::NamedTuple`       — kwargs pasados a `perturb(; ...)`
- `grid::NamedTuple`          — kwargs pasados a `construir_R(; ...)`. Ej:
                                 `(coords=:cartesian, x=-60:1:60, y=-60:1:60, z=-60:1:60)`
                                 `(coords=:cylindrical, r=0:5:160, theta=0:10:350, z=-60:5:60)`
- `out_dir::String`

`verbose=true` imprime una línea por iteración con los tiempos de cada paso.
Siempre imprime un resumen agregado al final (mean/std/min/max/total por paso),
y guarda esos agregados como `attrs` del HDF5.

Genera **ambos** outputs (B_grid y B_sens) por muestra en el mismo HDF5,
calculados sobre la misma config perturbada para garantizar consistencia
input ↔ output. Datasets en mT.

Convención de dimensiones (ver README):
- N = nº de muestras (datasets)
- I = nº de sensores             (índice i ∈ {1..I})
- J = nº de puntos de grilla     (índice j ∈ {1..J}), J = Nx·Ny·Nz

Layout del HDF5:
- `B_grid` :: (N, J, 3) Float32 — campo en cada punto de la grilla, mT.
              Si la grilla es un meshgrid regular, podés reshapearlo en el
              consumer (ej. NumPy: `B.reshape(N, Nx, Ny, Nz, 3)` con
              x-outer, z-inner).
- `B_sens` :: (N, I, 3) Float32 — campo en sensores, mT.
- `R_grid_xyz_mm` :: (J, 3) — posiciones xyz mm de los puntos de grilla.
- `sens_xyz_mm`   :: (I, 3) — posiciones xyz mm de los sensores.
- `grid_<axis>`   :: 1D — uno por cada eje de `config.grid` (ej. grid_x, grid_y, grid_z;
                          o grid_r, grid_theta, grid_z para cilíndrico).
- `attrs`         — coords del grid, params de perturb, timing summary.
"""
function generate_dataset(config; verbose::Bool = false)
    out_path = joinpath(config.out_dir, "$(config.name).h5")
    mkpath(config.out_dir)
    timer = StepTimer()

    # Construir R una sola vez (no cambia entre muestras) — también medido
    t0 = time_ns()
    R_grid = construir_R(; config.grid...)        # acepta cualquier kwargs de construir_R
    R_sens = R_sensores()
    record!(timer, :setup_R, time_ns() - t0)

    J = size(R_grid, 1)   # puntos de grilla (Nx·Ny·Nz aplanado)
    I = size(R_sens, 1)   # sensores

    # IMPORTANTE: HDF5.jl escribe arrays en column-major (vista Julia), entonces
    # h5py los ve transpuestos. Para que Python lea naturalmente `(N, J, 3)`,
    # declaramos las dims en Julia con orden inverso y escribimos transpuestos.
    # Vista Julia `(3, J, N)` ⇔ vista Python `(N, J, 3)`. Idem para los R.
    h5open(out_path, "w") do f
        ds_grid = create_dataset(f, "B_grid", Float32,
                                  ((3, J, config.n_samples),
                                   (3, J, -1));
                                  chunk = (3, J, 1))
        ds_sens = create_dataset(f, "B_sens", Float32,
                                  ((3, I, config.n_samples),
                                   (3, I, -1));
                                  chunk = (3, I, 1))
        f["R_grid_xyz_mm"] = permutedims(R_grid)   # (J,3) Julia → Python ve (J,3)
        f["sens_xyz_mm"]   = permutedims(R_sens)   # (I,3) Julia → Python ve (I,3)

        # Guardar cada eje del grid como dataset separado (grid_x, grid_y, grid_z,
        # o grid_r, grid_theta, grid_z dependiendo del sistema)
        for (k, v) in pairs(config.grid)
            k === :coords && continue
            f["grid_$(k)"] = collect(Float32, v)
        end

        attrs(f)["grid_coords"] = String(config.grid.coords)
        attrs(f)["sigma_deg"]   = config.perturb.sigma_deg
        attrs(f)["mu1"]         = config.perturb.mu1
        attrs(f)["sigma1"]      = config.perturb.sigma1
        attrs(f)["mu2"]         = config.perturb.mu2
        attrs(f)["sigma2"]      = config.perturb.sigma2
        attrs(f)["kind"]        = String(config.perturb.kind)
        attrs(f)["n_samples"]   = config.n_samples
        attrs(f)["seed_base"]   = config.seed_base

        for i in 1:config.n_samples
            tp = time_ns()
            P, M = perturb(; config.perturb...,
                            seed = config.seed_base + i)
            record!(timer, :perturb, time_ns() - tp)

            tg = time_ns()
            B_grid = B0(R_grid, P, M; timer = timer, prefix = :B0_grid)     # (J, 3) Tesla
            record!(timer, :B0_grid_total, time_ns() - tg)

            ts = time_ns()
            B_sens = B0(R_sens, P, M; timer = timer, prefix = :B0_sens)     # (I, 3) Tesla
            record!(timer, :B0_sens_total, time_ns() - ts)

            tw = time_ns()
            # Transponer (J,3)→(3,J) e (I,3)→(3,I) para que Python lea (N,J,3)/(N,I,3)
            ds_grid[:, :, i] = permutedims(B_grid .* 1000f0)   # mT
            ds_sens[:, :, i] = permutedims(B_sens .* 1000f0)   # mT
            record!(timer, :hdf5_write, time_ns() - tw)

            if verbose
                @printf "[%4d/%d] perturb=%5.1f  B0_grid=%6.1f (gpu=%4.1f K=%5.1f cpu=%4.1f free=%4.1f)  B0_sens=%5.1f (gpu=%4.1f K=%4.1f cpu=%4.1f free=%4.1f)  write=%6.1f  ms\n" i config.n_samples last_ms(timer, :perturb) last_ms(timer, :B0_grid_total) last_ms(timer, :B0_grid_to_gpu) last_ms(timer, :B0_grid_kernel) last_ms(timer, :B0_grid_to_cpu) last_ms(timer, :B0_grid_free) last_ms(timer, :B0_sens_total) last_ms(timer, :B0_sens_to_gpu) last_ms(timer, :B0_sens_kernel) last_ms(timer, :B0_sens_to_cpu) last_ms(timer, :B0_sens_free) last_ms(timer, :hdf5_write)
            end

            i % 50 == 0 && (GC.gc(); CUDA.reclaim())
        end

        # Persistir resumen de tiempos en attrs
        for (k, v) in summary_attrs(timer)
            attrs(f)[k] = v
        end
    end

    println("\ngenerate_dataset: $(config.n_samples) muestras → $(out_path)")
    report(timer)
    return out_path
end

"""
    generate_dataset_chunked(config; n_total, chunk_size, verbose=false,
                              skip_existing=true) -> Vector{String}

Genera `n_total` muestras repartidas en archivos HDF5 de a `chunk_size`
muestras cada uno (el último puede ser menor). Los seeds son contiguos
entre archivos: bit-exacto equivalente a un único `generate_dataset` con
`n_samples = n_total`.

Cada archivo se nombra `<config.name>_part<NN>.h5` con padding automático
según el total de chunks. Los seeds del chunk `c` (1-based) son
`config.seed_base + (c-1)*chunk_size + 1 .. config.seed_base + c*chunk_size`.

Parámetros
----------
- `n_total::Int`         — total de muestras a generar.
- `chunk_size::Int`      — muestras por archivo.
- `verbose::Bool`        — pasado a cada `generate_dataset`.
- `skip_existing::Bool`  — si `true` (default), salta chunks cuyo `.h5` ya
                           existe (resumible). Si `false`, los sobreescribe.

Retorna la lista de paths generados/encontrados (en orden).
"""
function generate_dataset_chunked(config;
        n_total::Int,
        chunk_size::Int,
        verbose::Bool       = false,
        skip_existing::Bool = true)

    n_total > 0   || error("n_total debe ser > 0")
    chunk_size > 0 || error("chunk_size debe ser > 0")

    n_chunks = cld(n_total, chunk_size)
    pad      = ndigits(n_chunks)
    paths    = String[]

    println("generate_dataset_chunked: $n_total muestras en $n_chunks archivos de hasta $chunk_size c/u")

    for c in 1:n_chunks
        chunk_n    = min(chunk_size, n_total - (c-1)*chunk_size)
        chunk_seed = config.seed_base + (c-1)*chunk_size
        chunk_name = "$(config.name)_part$(lpad(c, pad, '0'))"
        out_path   = joinpath(config.out_dir, "$(chunk_name).h5")

        println("\n>>> Chunk $c/$n_chunks: $(chunk_name) (seeds $(chunk_seed+1)..$(chunk_seed+chunk_n))")

        if skip_existing && isfile(out_path)
            println("    ✓ ya existe, salteando: $out_path")
            push!(paths, out_path)
            continue
        end

        chunk_config = merge(config, (
            name      = chunk_name,
            n_samples = chunk_n,
            seed_base = chunk_seed,
        ))
        push!(paths, generate_dataset(chunk_config; verbose = verbose))
    end

    println("\ngenerate_dataset_chunked: listo, $n_total muestras en $(length(paths)) archivos.")
    return paths
end

"""
    benchmark_dataset(config; warmup=1) -> StepTimer

Igual que `generate_dataset` pero **sin escribir HDF5**: corre el loop
completo y devuelve el `StepTimer` con todas las mediciones. Útil para
estimar cuánto va a tardar un experimento antes de lanzarlo.

`warmup` corre N iteraciones descartadas antes de medir, para incluir
JIT compile + primer kernel launch (que es notablemente más lento que
los siguientes).
"""
function benchmark_dataset(config; warmup::Int = 1)
    R_grid = construir_R(; config.grid...)
    R_sens = R_sensores()

    # Warmup (no medido)
    for i in 1:warmup
        P, M = perturb(; config.perturb..., seed = config.seed_base + i)
        B0(R_grid, P, M)
        B0(R_sens, P, M)
    end

    timer = StepTimer()
    for i in 1:config.n_samples
        tp = time_ns()
        P, M = perturb(; config.perturb..., seed = config.seed_base + i)
        record!(timer, :perturb, time_ns() - tp)

        tg = time_ns()
        B_grid = B0(R_grid, P, M; timer = timer, prefix = :B0_grid)
        record!(timer, :B0_grid_total, time_ns() - tg)

        ts = time_ns()
        B_sens = B0(R_sens, P, M; timer = timer, prefix = :B0_sens)
        record!(timer, :B0_sens_total, time_ns() - ts)

        i % 50 == 0 && (GC.gc(); CUDA.reclaim())
    end

    println("\nbenchmark_dataset: $(config.n_samples) muestras (warmup=$(warmup))")
    report(timer; title = "Benchmark generate_dataset")
    return timer
end

# Ejemplo de uso:
#
# # Cartesiano cúbico
# config = (
#     name      = "grid60_step2_sigma1_n100",
#     n_samples = 100, seed_base = 0,
#     perturb   = (kind=:both, sigma_deg=1f0,
#                  mu1=2.035f0, sigma1=0.1f0,
#                  mu2=8.48f0,  sigma2=0.85f0),
#     grid      = (coords=:cartesian, x=-60:2:60, y=-60:2:60, z=-60:2:60),
#     out_dir   = joinpath(@__DIR__, "data", "datasets"),
# )
# generate_dataset(config; verbose=true)
#
# # Cartesiano rectangular (asimétrico)
# grid = (coords=:cartesian, x=-100:10:100, y=-100:10:100, z=-220:10:220)
#
# # Cilíndrico
# grid = (coords=:cylindrical, r=0:5:160, theta=0:10:350, z=-60:5:60)
#
# # Estimar tiempo sin escribir disco:
# benchmark_dataset(config; warmup=2)
#
# # Splittear un experimento grande en chunks (resumible):
# generate_dataset_chunked(config; n_total=10_000, chunk_size=1_000)
# # → exp_part01.h5 .. exp_part10.h5
