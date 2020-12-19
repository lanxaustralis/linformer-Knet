using Knet
include("./operations.jl")
# TODO: tensor definition to stabilize fields
# TODO: must refer the linformer/attention all u need

# 1 - Embedding
struct Embed
    w::Any
end
Embed(vocabsize::Int, embedsize::Int) = Embed(param(embedsize, vocabsize))
(e::Embed)(x) = permutedims(e.w[:, x], (3, 1, 2)) .+ PE(size(x)[2], size(e.w)[1]) # returns tensor of the form T,E,B

function PE(inputsize::Int, embedsize::Int)
    PE = zeros(inputsize, embedsize)
    pos = collect(0:inputsize-1)

    for i = 1:Int(embedsize / 2)
        core = pos / 10^(5(2i - 2) / embedsize)
        PE[:, 2i-1] = sin.(core)
        PE[:, 2i] = cos.(core)
    end
    PE
end
# 2 - Dense
struct Dense
    w::Any
    b::Any
    σ::Function
end
Dense(inputsize::Int, outputsize::Int, σ::Function = identity) =
    Dense(param(outputsize, inputsize), param0(outputsize), σ)
(d::Dense)(x) = d.σ(d.w * mat(x, dims = 1) .+ d.b)

# 3 - LayerNorm
struct LayerNorm
    γ::Any
    β::Any
end
LayerNorm(layer_length::Int, layer_dim::Int) =
    LayerNorm(param(layer_length, layer_dim), param0(layer_dim))
(ln::LayerNorm)(x) = ln.γ * x .+ ln.β # TODO: ensure mat operation is not needed

# 4 - MH(L)A
struct MHA
    head_num::Int       # number of attention heads
    W_Q::Any
    W_K::Any
    W_V::Any
    W_O::Any
    E::Any              # Linformer : Linear Projection Matrix E
    F::Any              # Linformer : Linear Projection Matrix F
end

# Naive MHA
MHA(embed_dim::Int, head_num::Int = 1, hidden_dim::Int = Int(floor(embed_dim / head_num))) =
    MHA(
        head_num,
        param(embed_dim, hidden_dim),
        param(embed_dim, hidden_dim),
        param(embed_dim, hidden_dim),
        param(head_num * hidden_dim, embed_dim),
    )

# Linformer: MHLA
MHLA(
    embed_dim::Int,
    seq_length::Int,
    projected_dimension::Int,
    head_num::Int = 1,
    hidden_dim::Int = Int(floor(embed_dim / head_num)),
) = MHA(
    head_num,
    param(embed_dim, hidden_dim),
    param(embed_dim, hidden_dim),
    param(embed_dim, hidden_dim),
    param(head_num * hidden_dim, embed_dim),
    param(projected_dimension, seq_length),
    param(projected_dimension, seq_length),
)
