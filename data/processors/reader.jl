# I shall refer Comp 542 Natural Language Processing by Deniz Yüret @ Koç University, 2019
include("./tokenizer.jl")
abstract type AbstractReader end

# Valid for all iterators
Base.IteratorSize(::Type{AbstractReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{AbstractReader}) = Base.HasEltype()
Base.eltype(::Type{AbstractReader}) = Vector{Int}

struct TextReader <: AbstractReader
    file::String
    vocab::Vocab
end

word2ind(x; dict::Dict{String,Int} = Dict(" " => 1)) = get(dict, x, 1)

#Implementing the iterate function
function Base.iterate(r::TextReader, s = nothing)
    if s == nothing
        state = open(r.file)
        Base.iterate(r, state)
    else
        eof(s) && (close(s); return nothing)
        sentence =
            r.vocab.tokenizer(strip(lowercase(readline(s))), [' '], keepempty = false)
        return (word2ind.(sentence, dict = r.vocab.w2i), s)
    end
end
