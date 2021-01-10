# I shall refer Comp 542 Natural Language Processing by Deniz Yüret @ Koç University, 2019
include("./tokenizer_wrapper.jl")

struct TextReader
    file::String
    bert_tokenizer::BertTokenizer
end

#Implementing the iterate function
function Base.iterate(r::TextReader, s = nothing)
    if s == nothing
        state = open(r.file)
        Base.iterate(r, state)
    else
        eof(s) && (close(s); return nothing)
        return (r.bert_tokenizer.tokenizer.encode(readline(s)).ids, s)
    end
end

# Iterator settings : Must stay at the end of the file
Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}
