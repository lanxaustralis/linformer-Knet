def setup_bert_tokenizer(
    vocab_size,
    unk_token,
    sep_token,
    pad_token,
    cls_token,
    mask_token,
    ):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers import normalizers
    from tokenizers.normalizers import Lowercase,NFD,StripAccents
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import WordPieceTrainer

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

def train_tokenizer(tokenizer, trn_files):
    tokenizer.train(tokenizer.trainer, trn_files)

def save_tokenizer(tokenizer, save_path, tokenizer_name, unk_token):
    files = tokenizer.model.save(save_path, tokenizer_name)
    tokenizer.model = type(tokenizer.model).from_file(*files, unk_token = unk_token)
    tokenizer.save(tokenizer_name)

def unsafe_load_tokenizer(load_path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(load_path)

def unsafe_setup_bert_fixed_tokenizer(model_config, fast=False):
    model_class = None
    if fast:
        from transformers import BertTokenizerFast
        model_class = BertTokenizerFast
    else:
        from transformers import BertTokenizer
        model_class = BertTokenizer

    return model_class.from_pretrained(model_config)
