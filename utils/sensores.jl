using DelimitedFiles

"""
    R_sensores(csv_path = "data/coordenadas_sensores.csv") -> Matrix{Float32}

Lee la geometría de los sensores del CSV (fuente de verdad del hardware) y
devuelve la matriz de evaluación `R` shape `[n, 3]` en xyz mm.

El CSV tiene columnas `(r_cm, θ_deg, z_cm)`. Pre-procesa unidades y offset, y
delega a `construir_R(:cylindrical, points)` definida en `B0.jl`.

Si en el futuro la geometría de los sensores cambia (más sensores, no
cilíndrica, etc.), se actualiza este archivo y/o el CSV; el resto del pipeline
no necesita cambios.
"""
function R_sensores(csv_path::AbstractString = joinpath(@__DIR__, "..", "data", "coordenadas_sensores.csv"))
    coords_csv = readdlm(csv_path, ',', Float32)               # [n, 3] (r_cm, θ_deg, z_cm)
    points_mm  = hcat(coords_csv[:, 1] .* 10f0,                # r: cm → mm
                      coords_csv[:, 2],                         # θ: grados (sin cambio)
                      (coords_csv[:, 3] .- 22f0) .* 10f0)       # z: offset cm + cm → mm
    return construir_R(:cylindrical, points_mm)
end
