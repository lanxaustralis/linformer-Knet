using Statistics: mean, std
const ϵ = 1e-05
# Normalization
function normalize(x, dims...)
    μ = mean(x, dims = dims)
    σ = std(x, mean = μ, dims = dims) .+ ϵ
    (x .- μ) ./ σ
end

isNormal(x) = isapprox(x, normalize(x))

# 3D tensor multiplied with 2D, resulting 3D
two2three(A, B) = (
    A == 1 ? B :
    A == 0 ? 0 :
    reshape(reshape(A, size(A)[1:2]) * mat(B, dims = 1), (:, size(B)[2:end]...))
)

# 2D tensor multiplied with 3D, resulting 3D
three2two(A, B) = (
    A == 1 ? B :
    A == 0 ? 0 : reshape(reshape(A, :, size(A, 2)) * B, (:, size(B, 2), size(A, 3)))
)

# 1D broadcasted to 3D, resulting 3D
one2three(A, B; f::Function = *) = (
    A == 1 ? B :
    A == 0 ? 0 :
    reshape(
        collect(Iterators.flatten([f.(A[i], B[:, :, i]) for i = 1:size(B, 3)])),
        size(B),
    )
)
