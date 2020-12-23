# The very first iterator is naive LM iterator - without any mask or projection of data (i.e. as in Machine Translation or Sentiment Analysis Tasks)
# I shall refer Comp 542 Natural Language Processing by Deniz Yüret @ Koç University, 2019
include("./reader.jl")
abstract type AbstractIterator end

Base.IteratorSize(::Type{AbstractIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{AbstractIterator}) = Base.HasEltype()
Base.eltype(::Type{AbstractIterator}) = Matrix{Int}

struct NaiveLMData <: AbstractIterator
    src::TextReader
    batchsize::Int
    maxlength::Int
    bucketwidth::Int
    buckets::Any
end

function NaiveLMData(
    src::TextReader;
    batchsize = 64,
    maxlength = 200,
    bucketwidth = 10,
)
    numbuckets = min(128, maxlength ÷ bucketwidth)
    buckets = [[] for i = 1:numbuckets]
    NaiveLMData(src, batchsize, maxlength, bucketwidth, buckets)
end

# Even if it is defined considering abstract type, each iterator must have its own unique iterate function
function Base.iterate(d::NaiveLMData, state = nothing)
    if state == nothing
        for b in d.buckets
            empty!(b)
        end
    end
    bucket, ibucket = nothing, nothing
    while true
        iter = (state === nothing ? iterate(d.src) : iterate(d.src, state))
        if iter === nothing
            ibucket = findfirst(x -> !isempty(x), d.buckets)
            bucket = (ibucket === nothing ? nothing : d.buckets[ibucket])
            break
        else
            sent, state = iter
            length(sent) > d.maxlength - 1 || length(sent) == 0 && continue # skipping improper sentence
            ibucket = min(1 + (length(sent) - 1) ÷ d.bucketwidth, length(d.buckets))
            bucket = d.buckets[ibucket]
            push!(bucket, sent)
            length(bucket) === d.batchsize && break # The last batch of the iterator
        end
    end
    bucket === nothing && return nothing
    batch = fill(d.src.vocab.eos, length(bucket), d.maxlength)

    for i = 1:length(bucket)
        batch[i, 1:length(bucket[i])] = bucket[i]
    end

    empty!(bucket)
    batch, state
end
