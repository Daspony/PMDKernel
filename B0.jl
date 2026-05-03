using NPZ
using CUDA
include("kernel.jl")
isdefined(@__MODULE__, :NOMINAL_SCALE_1) || include("utils/calibracion.jl")
isdefined(@__MODULE__, :StepTimer)        || include("utils/timer.jl")

"""
    B0(R, P, M; threads=256, timer=nothing, prefix=:B0) -> Matrix{Float32}

Función pura: calcula el campo dipolar en los puntos `R` dadas las posiciones
`P` y momentos `M` de los imanes.

Parámetros
----------
- `R` :: Matrix{Float32}, shape `[n, 3]`, posiciones en **mm** (xyz).
- `P` :: Matrix{Float32}, shape `[3, m]`, posiciones de imanes en **mm**.
- `M` :: Matrix{Float32}, shape `[3, m]`, momentos en **A·m²**.
- `threads` :: hilos CUDA por bloque (default = 256).
- `timer` :: si se pasa un `StepTimer`, registra el desglose interno con keys
  `<prefix>_to_gpu`, `<prefix>_kernel`, `<prefix>_to_cpu`, `<prefix>_free`.
  Cuando es `nothing` (default), no se hacen sincronizaciones extra.
- `prefix` :: prefijo para las keys del timer (ej. `:B0_grid`, `:B0_sens`).

Retorna `B::Matrix{Float32}` shape `[n, 3]` en **Tesla**.

Sin efectos secundarios: no carga archivos, no escribe archivos, no abre GUI.
"""
function B0(R::AbstractMatrix, P::AbstractMatrix, M::AbstractMatrix;
        threads::Int = 256,
        timer = nothing,
        prefix::Symbol = :B0)
    n = size(R, 1)
    m = size(P, 2)

    # Host prep + H2D transfer + alloc B_gpu
    t = time_ns()
    R_gpu = CuArray(Float32.(R) .* 0.001f0)
    P_gpu = CuArray(Float32.(P) .* 0.001f0)
    M_gpu = CuArray(Float32.(M))
    B_gpu = CUDA.zeros(Float32, n, 3)
    if timer !== nothing
        CUDA.synchronize()
        record!(timer, Symbol(prefix, :_to_gpu), time_ns() - t)
    end

    # Kernel
    blocks = cld(n, threads)
    t = time_ns()
    @cuda threads=threads blocks=blocks _Bnu!(R_gpu, P_gpu, M_gpu, B_gpu, n, m)
    if timer !== nothing
        CUDA.synchronize()
        record!(timer, Symbol(prefix, :_kernel), time_ns() - t)
    end

    # D2H
    t = time_ns()
    B_cpu = Array(B_gpu)
    timer !== nothing && record!(timer, Symbol(prefix, :_to_cpu), time_ns() - t)

    # Free
    t = time_ns()
    CUDA.unsafe_free!(R_gpu)
    CUDA.unsafe_free!(P_gpu)
    CUDA.unsafe_free!(M_gpu)
    CUDA.unsafe_free!(B_gpu)
    timer !== nothing && record!(timer, Symbol(prefix, :_free), time_ns() - t)

    return B_cpu
end

"""
    cargar_imanes(data_path; raw=true) -> (P, M)

Lee un NPZ con formato B0 (`array1..array4`) y devuelve `(P, M)`:
- `P` :: Matrix{Float32} `[3, m]` posiciones en mm.
- `M` :: Matrix{Float32} `[3, m]` momentos en A·m².

Si `raw=true` (default), aplica calibración nominal a `array2`/`array4`.
Si `raw=false`, los toma como vienen (caso típico: salida de `perturb`).
"""
function cargar_imanes(data_path::AbstractString; raw::Bool = true)
    data = npzread(data_path)
    P = Float32.(hcat(data["array1"], data["array3"]))
    if raw
        M = Float32.(hcat(data["array2"] .* NOMINAL_SCALE_1,
                          data["array4"] .* NOMINAL_SCALE_2))
    else
        M = Float32.(hcat(data["array2"], data["array4"]))
    end
    return P, M
end

# --- construir_R: builder unificado de R en xyz mm --------------------------
#
# Convención cilíndrica del proyecto: x = -r·sin(θ), y = r·cos(θ)
# (preserva la orientación "+x crece derecha→izquierda" del resonador).

@inline function _cyl_to_xyz(r::Float32, theta_deg::Float32, z::Float32)
    return -r * sind(theta_deg), r * cosd(theta_deg), z
end

"""
    construir_R(; coords=:cartesian, kwargs...) -> Matrix{Float32}

Construye una grilla de evaluación (meshgrid) en xyz mm.

- `coords = :cartesian` + kwargs `x, y, z` (Ranges/Vectors en mm)
  → producto cartesiano `R[Nx·Ny·Nz, 3]`.
- `coords = :cylindrical` + kwargs `r, theta, z` (mm, °, mm)
  → producto cilíndrico → `R[Nr·Nθ·Nz, 3]` en xyz mm.
"""
function construir_R(; coords::Symbol = :cartesian, kwargs...)
    if coords === :cartesian
        x = kwargs[:x]; y = kwargs[:y]; z = kwargs[:z]
        Nx = length(x); Ny = length(y); Nz = length(z)
        R = Matrix{Float32}(undef, Nx*Ny*Nz, 3)
        idx = 1
        for xi in x, yi in y, zi in z
            R[idx, 1] = Float32(xi)
            R[idx, 2] = Float32(yi)
            R[idx, 3] = Float32(zi)
            idx += 1
        end
        return R
    elseif coords === :cylindrical
        r = kwargs[:r]; theta = kwargs[:theta]; z = kwargs[:z]
        Nr = length(r); Nt = length(theta); Nz = length(z)
        R = Matrix{Float32}(undef, Nr*Nt*Nz, 3)
        idx = 1
        for ri in r, ti in theta, zi in z
            xi, yi, zi2 = _cyl_to_xyz(Float32(ri), Float32(ti), Float32(zi))
            R[idx, 1] = xi
            R[idx, 2] = yi
            R[idx, 3] = zi2
            idx += 1
        end
        return R
    else
        error("construir_R: coords debe ser :cartesian o :cylindrical (recibido $(coords))")
    end
end

"""
    construir_R(coords, points::AbstractMatrix) -> Matrix{Float32}

Convierte una lista de puntos `points` shape `[n, 3]` al sistema xyz mm.

- `coords = :cartesian`   → identidad (cada fila ya es xyz mm).
- `coords = :cylindrical` → cada fila `(r_mm, θ_deg, z_mm)` → xyz mm.
"""
function construir_R(coords::Symbol, points::AbstractMatrix)
    n = size(points, 1)
    @assert size(points, 2) == 3 "construir_R: points debe ser [n, 3]"
    if coords === :cartesian
        return Matrix{Float32}(points)
    elseif coords === :cylindrical
        R = Matrix{Float32}(undef, n, 3)
        @inbounds for i in 1:n
            xi, yi, zi = _cyl_to_xyz(Float32(points[i, 1]),
                                      Float32(points[i, 2]),
                                      Float32(points[i, 3]))
            R[i, 1] = xi
            R[i, 2] = yi
            R[i, 3] = zi
        end
        return R
    else
        error("construir_R: coords debe ser :cartesian o :cylindrical (recibido $(coords))")
    end
end
