
# from https://github.com/goombalab/hnet/blob/main/hnet/utils/tokenizers.py




import numpy as np


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs: list[str], add_bos: bool = False, add_eos: bool = False, **kwargs) -> list[dict[str, np.ndarray]]:
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens: np.ndarray | list[int], **kwargs) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)
    










# --------------------------------------




# bpe tokenizer



import sentencepiece as spm

class BPETokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()
    
    def encode(self, text: str, add_bos=True, add_eos=True):
        tokens = self.sp.encode(text, out_type=int)
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens
    
    def decode(self, token_ids: list[int]):
        return self.sp.decode(token_ids)
    
    def vocab_size(self):
        return self.sp.get_piece_size()
