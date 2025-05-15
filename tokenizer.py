import torch


class CharTokenizer:
    def __init__(self) -> None:
        # self.vocab = [
        #     "\n", " ", "!", "$", "&", "'", ",", "-", ".", "3", ":", ";", "?", "A", "B", "C", "D", 
        #     "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
        #     "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", 
        #     "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
        # ]
        self.vocab = [
            '\n', ' ', '!', '(', ')', '*', ',', '-', '.', '1', '3', '4', '5', '9', ':', ';', '?', 
            'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 
            'V', 'X', 'Y', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 
            'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'Á', 'Â', 'Ê', 'Í', 'Ô', 'à', 'á', 'â', 'ã', 
            'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă', 'Đ', 'đ', 'ĩ', 'ũ', 
            'ơ', 'ư', 'ạ', 'ả', 'Ấ', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ẹ', 'ẻ', 
            'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ', 'ỏ', 'ố', 'Ồ', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 
            'Ờ', 'ờ', 'Ở', 'ở', 'ỡ', 'ợ', 'ụ', 'ủ', 'ứ', 'ừ', 'Ử', 'ử', 'ữ', 'ự', 'ỷ', 'ỹ', '“', 
            '”'
        ]
        self.vocab_len = len(self.vocab)
        self.ctoi = {c: i for i, c in enumerate(self.vocab)}
        self.itoc = {i: c for i, c in enumerate(self.vocab)}

    def encode(self, x: str) -> torch.Tensor:
        """
        Encode a string into a tensor of integers.
        """
        encoded = torch.tensor([self.ctoi[c] for c in x], dtype=torch.long)
        return encoded.unsqueeze(0) # add batch dimension
    
    def decode(self, x: torch.Tensor) -> str:
        """
        Decode a tensor of integers into a string.
        """
        x = x.squeeze(0).numpy() # remove batch dimension
        decoded = "".join([self.itoc[i] for i in x])
        return decoded
    
    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary.
        """
        return self.vocab_len


if __name__ == "__main__":
    DATA_PATH = "/home/dc/self_studies/baby_gpt/data/xuandieutho.txt"

    with open(DATA_PATH, "r") as _f:
        data = _f.read()

    print(len(data))
    print(sorted(list(set(data))))
    # data = data[:333]
    # print(data)

    # tokenizer = CharTokenizer()

    # print("---")
    # encoded = tokenizer.encode(data)
    # # print(encoded)

    # decoded = tokenizer.decode(encoded)
    # print(decoded)
