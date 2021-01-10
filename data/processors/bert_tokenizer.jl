using PyCall

py"""
# Ensure tokenizers package is already imported
import os
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase,NFD,StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

def setup_tokenizer(
    vocab_size,
    unk_token,
    sep_token,
    pad_token,
    cls_token,
    mask_token,
    ):

    tokenizer = Tokenizer(WordPiece())
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(),
            StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single=cls_token
            + ' $A ' + sep_token, pair=cls_token + ' $A ' + sep_token
            + ' $B:1 ' + sep_token + ':1', special_tokens=[(cls_token,
            1), (sep_token, 2)])
    tokenizer.trainer = WordPieceTrainer(vocab_size=vocab_size,
            special_tokens=[unk_token, cls_token, sep_token, pad_token,
            mask_token])
    return tokenizer

def train_tokenizer(tokenizer,trn_files):
    tokenizer.train(tokenizer.trainer,trn_files)

def save_tokenizer(tokenizer, path, tokenizer_name, unk_token):
    files = tokenizer.model.save(path,tokenizer_name)
    tokenizer.model = type(tokenizer.model).from_file(*files, unk_token = unk_token)
    tokenizer.save(tokenizer_name)

def load_tokenizer(path):
    return Tokenizer.from_file(path)
"""

struct BertTokenizer
    tokenizer::PyObject
    unk_token::String
    sep_token::String
    pad_token::String
    cls_token::String
    mask_token::String
end

# Define BertTokenizer from scratch
BertTokenizer(
    vocab_size::Int,
    unk_token = "[UNK]",
    sep_token = "[SEP]",
    pad_token = "[PAD]",
    cls_token = "[CLS]",
    mask_token = "[MASK]",
) = BertTokenizer(
    py"setup_tokenizer"(
        vocab_size,
        unk_token,
        sep_token,
        pad_token,
        cls_token,
        mask_token,
    ),
    unk_token,
    sep_token,
    pad_token,
    cls_token,
    mask_token,
)

# Load BertTokenizer from predefined tokenizer
# TODO: What to do in the case of special token re=assignment/mismatch?
BertTokenizer(
    tokenizer_path::String,
    unk_token = "[UNK]",
    sep_token = "[SEP]",
    pad_token = "[PAD]",
    cls_token = "[CLS]",
    mask_token = "[MASK]",
) = BertTokenizer(
    py"load_tokenizer"(tokenizer_path),
    unk_token,
    sep_token,
    pad_token,
    cls_token,
    mask_token,
)

# Perform training operations over provided files
(bert_tokenizer::BertTokenizer)(trn_files::Vector{String}) =
    py"train_tokenizer"(bert_tokenizer.tokenizer, trn_files)

# Must save and reload the tokenizer after training
(bert_tokenizer::BertTokenizer)(
    save_path::String,
    tokenizer_name::String,
    unk_token = "[UNK]",
) = py"save_tokenizer"(
    bert_tokenizer.tokenizer,
    save_path,
    tokenizer_name,
    unk_token,
)
