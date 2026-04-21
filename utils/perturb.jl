using NPZ
using Distributions
using Random

# Factores nominales aplicados a los vectores raw de B0.npz
const NOMINAL_SCALE_1 = 2.035f0   # para array2 (1936 imanes)
const NOMINAL_SCALE_2 = 3.051f0   # para array4 (384 imanes)

# --- Helpers internos ---

function rotate_xy(x::Float32, y::Float32, delta_rad::Float32)
    c = cos(delta_rad); s = sin(delta_rad)
    return x*c - y*s, x*s + y*c
end

# Rota en el plano XY los vectores de momento de cada imán.
# delta ~ Normal(0, sigma_deg) grados, independiente por columna.
function apply_rotation!(mom::Matrix{Float32}, rng, dist)
    for j in axes(mom, 2)
        delta = Float32(deg2rad(rand(rng, dist)))
        mom[1,j], mom[2,j] = rotate_xy(mom[1,j], mom[2,j], delta)
    end
end

# Muestrea un factor de escala por imán de Normal(mu, sigma). Asume mu > 0.
function sample_scales(rng, mu::Float32, sigma::Float32, m::Int)::Vector{Float32}
    scales = Float32.(rand(rng, Normal(mu, sigma), m))
    @assert all(scales .> 0f0) "Escala negativa muestreada — verifica mu y sigma"
    return scales
end

"""
    perturb(; kind, sigma_deg, mu1, sigma1, mu2, sigma2, seed, out_path) -> String

Genera un NPZ con los imanes perturbados (momentos ya escalados a unidades SI).

Parámetros
----------
- `kind`       : `:rotation` | `:magnitude` | `:both` (default)
- `sigma_deg`  : desviación angular (grados) para la rotación XY (default = 1.0)
- `mu1,sigma1` : media y sigma para la magnitud del array2 (default = 2.035 ± 0.1)
- `mu2,sigma2` : media y sigma para la magnitud del array4 (default = 3.051 ± 0.3)
- `seed`       : semilla RNG (default = nothing → aleatorio)
- `out_path`   : ruta del NPZ de salida (requerido)

Convención: los momentos guardados SIEMPRE están escalados — `simular_disco`
los lee tal cual, sin re-escalar.
"""
function perturb(; kind::Symbol     = :both,
        sigma_deg::Float32 = 1f0,
        mu1::Float32 = 2.035f0, sigma1::Float32 = 0.1f0,
        mu2::Float32 = 3.051f0, sigma2::Float32 = 0.3f0,
        seed              = nothing,
        out_path::String)

    kind in (:rotation, :magnitude, :both) ||
        error("kind debe ser :rotation, :magnitude o :both; recibido $(kind)")

    rng  = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    data = npzread(joinpath(@__DIR__, "..", "data", "B0.npz"))
    pos1 = Float32.(data["array1"]); pos2 = Float32.(data["array3"])
    mom1 = Float32.(data["array2"]) .* NOMINAL_SCALE_1
    mom2 = Float32.(data["array4"]) .* NOMINAL_SCALE_2

    if kind in (:magnitude, :both)
        # Los momentos ya están multiplicados por NOMINAL_SCALE_*, por eso el
        # factor que aplicamos debe centrarse en mu/NOMINAL para que el resultado
        # neto tenga media mu, no mu * NOMINAL.
        mom1 .*= sample_scales(rng, mu1 / NOMINAL_SCALE_1,
                                sigma1 / NOMINAL_SCALE_1, size(mom1, 2))'
        mom2 .*= sample_scales(rng, mu2 / NOMINAL_SCALE_2,
                                sigma2 / NOMINAL_SCALE_2, size(mom2, 2))'
    end
    if kind in (:rotation, :both)
        dist = Normal(0, sigma_deg)
        apply_rotation!(mom1, rng, dist)
        apply_rotation!(mom2, rng, dist)
    end

    npzwrite(out_path, Dict("array1" => pos1, "array2" => mom1,
                             "array3" => pos2, "array4" => mom2))
    return out_path
end

"""
    perturb_batch(n; kind, seed_base, out_dir, name_prefix, ...) -> Vector{String}

Genera `n` NPZ perturbados con seeds `seed_base+1 .. seed_base+n`. Los paths
quedan `<out_dir>/<name_prefix>_<kind>_seed<i>.npz`.

Ejemplo
-------
    paths = perturb_batch(10; kind=:both, sigma_deg=1f0)
"""
function perturb_batch(n::Int;
        kind::Symbol        = :both,
        seed_base::Int      = 0,
        out_dir::String     = joinpath(@__DIR__, "..", "data", "perturbed"),
        name_prefix::String = "B0",
        sigma_deg::Float32  = 1f0,
        mu1::Float32 = 2.035f0, sigma1::Float32 = 0.1f0,
        mu2::Float32 = 3.051f0, sigma2::Float32 = 0.3f0)

    mkpath(out_dir)
    paths = String[]
    for i in 1:n
        seed = seed_base + i
        out_path = joinpath(out_dir, "$(name_prefix)_$(kind)_seed$(seed).npz")
        perturb(; kind=kind, sigma_deg=sigma_deg,
                 mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2,
                 seed=seed, out_path=out_path)
        push!(paths, out_path)
    end
    println("perturb_batch: $(n) NPZ en $(out_dir) " *
            "(kind=$(kind), seeds $(seed_base+1)..$(seed_base+n))")
    return paths
end
