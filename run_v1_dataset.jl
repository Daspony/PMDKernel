include(joinpath(@__DIR__, "generate_dataset.jl"))

config = (
    name      = "v1_xy100_z225_step10_n5000",
    n_samples = 5000,
    seed_base = 0,
    perturb   = (kind=:both, sigma_deg=1f0,
                 mu1=2.035f0, sigma1=0.1f0,
                 mu2=8.48f0,  sigma2=0.85f0),
    grid      = (coords=:cartesian, x=-100:10:100, y=-100:10:100, z=-225:10:225),
    out_dir   = joinpath(@__DIR__, "data", "datasets"),
)

generate_dataset(config; verbose=true)
