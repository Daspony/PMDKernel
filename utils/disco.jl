using CUDA
using NPZ
using DelimitedFiles

# Uso típico (desde el REPL, en el root del proyecto):
#   const BATCH_M = 64
#   include("PMDKernel/kernel.jl")
#   include("PMDKernel/utils/perturb.jl")
#   include("PMDKernel/utils/disco.jl")
#
#   paths = perturb_batch(10; kind=:both, sigma_deg=1f0)
#   simular_disco_batch(0f0, paths;
#       out_path="PMDKernel/data/simulaciones/disco_z0_both_n10.npz")

"""
    simular_disco(z_mm; data_path, xy_step, grid_range, thickness, threads) -> NamedTuple

Computa el campo B en (a) una grilla XY en el plano z = `z_mm`, y (b) los
sensores cuya z cae en el disco `|z_sensor − z_mm| ≤ thickness/2`.

**No escribe archivos.** Devuelve los arrays para que el caller los apile o
los serialice como prefiera.

Parámetros
----------
- `z_mm`       : posición Z (mm) del plano del disco
- `data_path`  : ruta al NPZ de imanes (momentos ya escalados — como produce `perturb`)
- `xy_step`    : paso de la grilla XY (mm, default = 10)
- `grid_range` : rango (min, max) XY (mm, default = (-150, 150))
- `thickness`  : grosor del disco en Z (mm, default = 1)
- `threads`    : hilos CUDA por bloque (default = 256)

Retorna `(B_grid, B_sens, grid_xy, sens_xyz, sens_idx)`:
- `B_grid`  :: Array{Float32,3}   (Nx, Ny, 3)   mT
- `B_sens`  :: Matrix{Float32}    (K,  3)       mT
- `grid_xy` :: Array{Float32,3}   (Nx, Ny, 2)   mm
- `sens_xyz`:: Matrix{Float32}    (K,  3)       mm
- `sens_idx`:: Vector{Int32}      (K,)          índice 1-based en el CSV
"""
function simular_disco(z_mm::Real;
        data_path::String,
        xy_step::Real    = 10f0,
        grid_range::Tuple{<:Real,<:Real} = (-150f0, 150f0),
        thickness::Real  = 1f0,
        threads::Int     = 256)

    z_disco = Float32(z_mm)
    half_t  = Float32(thickness) / 2f0

    # --- Sensores (CSV cilíndricas cm, °, cm → cartesianas mm) ---
    coords = readdlm(joinpath(@__DIR__, "..", "data", "coordenadas_sensores.csv"), ',', Float32)
    n_tot  = size(coords, 1)
    sens_all = Matrix{Float32}(undef, n_tot, 3)
    for i in 1:n_tot
        r, θ = coords[i,1], coords[i,2]
        sens_all[i,1] = r * -sind(θ) * 10f0
        sens_all[i,2] = r *  cosd(θ) * 10f0
        sens_all[i,3] = (coords[i,3] - 22f0) * 10f0
    end
    mask     = abs.(sens_all[:,3] .- z_disco) .<= half_t
    sens_xyz = sens_all[mask, :]
    sens_idx = Int32.(findall(mask))
    K        = size(sens_xyz, 1)

    # --- Grilla XY ---
    gx = range(Float32(grid_range[1]); stop=Float32(grid_range[2]), step=Float32(xy_step))
    Nx = length(gx); Ny = Nx
    R_grid  = Matrix{Float32}(undef, Nx*Ny, 3)
    grid_xy = Array{Float32,3}(undef, Nx, Ny, 2)
    idx = 1
    for (ix, xi) in enumerate(gx), (iy, yi) in enumerate(gx)
        R_grid[idx, 1] = xi; R_grid[idx, 2] = yi; R_grid[idx, 3] = z_disco
        grid_xy[ix, iy, 1] = xi
        grid_xy[ix, iy, 2] = yi
        idx += 1
    end
    n_grid = Nx * Ny

    # --- Imanes (los momentos ya vienen escalados desde `perturb`) ---
    data  = npzread(data_path)
    P_cpu = Float32.(hcat(data["array1"], data["array3"]))
    M_cpu = Float32.(hcat(data["array2"], data["array4"]))
    m     = size(P_cpu, 2)

    # --- Kernel ---
    R_all = vcat(R_grid, sens_xyz)
    n_all = size(R_all, 1)
    R = CuArray(R_all .* 0.001f0)
    P = CuArray(P_cpu .* 0.001f0)
    M = CuArray(M_cpu)
    B = CuArray(zeros(Float32, n_all, 3))
    blocks = cld(n_all, threads)
    shmem  = 6 * BATCH_M * sizeof(Float32)
    @cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(R, P, M, B, n_all, m)

    B_all  = Array(B) .* 1000f0   # → mT
    B_grid = reshape(B_all[1:n_grid, :], Nx, Ny, 3)
    B_sens = K > 0 ? B_all[n_grid+1:end, :] : zeros(Float32, 0, 3)

    return (; B_grid, B_sens, grid_xy, sens_xyz, sens_idx)
end

"""
    simular_disco_batch(z_mm, perturbed_paths; xy_step, grid_range, thickness, threads, out_path) -> String

Corre `simular_disco` por cada path, apila los resultados en arrays N-dim y
escribe UN solo NPZ consolidado en `out_path`.

Formato del NPZ (mínimo necesario)
-----------------------------------
- `B_grid`   : Float32 (N, Nx, Ny, 3)   mT   — input de la red
- `B_sens`   : Float32 (N, K,  3)       mT   — target de la red
- `sens_xyz` : Float32 (K,  3)          mm   — posiciones de sensores (evita releer CSV)
- `z_disco`  : Float32 (escalar)        mm   — plano del disco

La grilla XY se reconstruye en el consumidor con `xy_step` y `grid_range`.
"""
function simular_disco_batch(z_mm::Real, perturbed_paths::Vector{String};
        xy_step::Real   = 10f0,
        grid_range::Tuple{<:Real,<:Real} = (-150f0, 150f0),
        thickness::Real = 1f0,
        threads::Int    = 256,
        out_path::String)

    n = length(perturbed_paths)
    n == 0 && error("perturbed_paths vacío")

    s0 = simular_disco(z_mm; data_path=perturbed_paths[1],
                       xy_step=xy_step, grid_range=grid_range,
                       thickness=thickness, threads=threads)
    Nx, Ny = size(s0.B_grid, 1), size(s0.B_grid, 2)
    K      = size(s0.sens_xyz, 1)

    B_grid_all = Array{Float32,4}(undef, n, Nx, Ny, 3)
    B_sens_all = Array{Float32,3}(undef, n, K,  3)

    B_grid_all[1, :, :, :] = s0.B_grid
    B_sens_all[1, :, :]    = s0.B_sens

    for i in 2:n
        s = simular_disco(z_mm; data_path=perturbed_paths[i],
                          xy_step=xy_step, grid_range=grid_range,
                          thickness=thickness, threads=threads)
        size(s.B_grid)   == size(s0.B_grid)   || error("grilla inconsistente en $(perturbed_paths[i])")
        size(s.sens_xyz) == size(s0.sens_xyz) || error("sensores inconsistentes en $(perturbed_paths[i])")
        B_grid_all[i, :, :, :] = s.B_grid
        B_sens_all[i, :, :]    = s.B_sens
    end

    mkpath(dirname(out_path))
    npzwrite(out_path, Dict(
        "B_grid"   => B_grid_all,
        "B_sens"   => B_sens_all,
        "sens_xyz" => s0.sens_xyz,
        "z_disco"  => Float32(z_mm),
    ))
    println("simular_disco_batch: $(n) muestras → $(out_path)  (grilla $(Nx)×$(Ny), K=$(K))")
    return out_path
end
