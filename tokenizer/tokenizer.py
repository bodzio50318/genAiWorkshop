import os


class BasicTokenizer:
    firstSpecialToken=256
    
    def __init__(self, text, vocab_size):
        self.text = text
        self.vocab_size = vocab_size
        self.merges = {}
        bytearrayText = bytearray(text.encode("utf-8"))
        self._train()

    def _train(self, verbose=False):
        print("Training tokenizer")
        while len(self.merges) < self.vocab_size:
            pair = self._getStats()[0][0]
            if verbose:
                print("Merging", pair)
            self._merge(pair)

    def _merge(self, pair):
        self.merges[pair] = BasicTokenizer.firstSpecialToken + len(self.merges)
        new_text = []
        i = 0
        while i < len(self.text):
            if i + 1 < len(self.text) and (self.text[i], self.text[i + 1]) == pair:
                new_text.append(pair)
                i += 2
            else:
                new_text.append(self.text[i])
                i += 1
        self.text = new_text

    def _getStats(self):
        stats = {}
        for i in range(len(self.text) - 1):
            pair = (self.text[i], self.text[i + 1])
            if pair in stats:
                stats[pair] += 1
            else:
                stats[pair] = 1
        return sorted(stats.items(), key=lambda x: x[1], reverse=True)
# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(current_dir, "pyrenees.txt")
# Load the input
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

basicTokenizer = BasicTokenizer(text, 500)

print(basicTokenizer.merges)
