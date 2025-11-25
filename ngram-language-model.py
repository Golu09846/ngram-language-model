# ngram_next_word.py
from collections import Counter, defaultdict
import math, re
from typing import List, Tuple, Dict

BOS = "<s>"
EOS = "</s>"

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+|[.,!?;:]", text.lower())

def make_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

class NgramLM:
    def __init__(self, n: int = 3, alpha: float = 0.5):
        self.n = n
        self.alpha = alpha
        self.vocab = set()
        self.ngram_counts: Dict[int, Counter] = defaultdict(Counter)

    def fit(self, corpus: List[str]) -> None:
        for line in corpus:
            toks = [BOS] * (self.n - 1) + tokenize(line) + [EOS]
            self.vocab.update(toks)
            for k in range(1, self.n + 1):
                for ng in make_ngrams(toks, k):
                    self.ngram_counts[k][ng] += 1
        self.V = len(self.vocab)

    def _conditional_logprob(self, word: str, history: Tuple[str, ...]) -> float:
        k = len(history) + 1
        ngram = history + (word,)
        num = self.ngram_counts[k][ngram] + self.alpha
        if k == 1:
            total_unigrams = sum(self.ngram_counts[1].values())
            den = total_unigrams + self.alpha * self.V
        else:
            hist_count = self.ngram_counts[k-1][history]
            den = hist_count + self.alpha * self.V
        return math.log(num) - math.log(den)

    def next_word_probs(self, context: List[str]) -> List[Tuple[str, float]]:
        hist = [BOS] * max(0, self.n - 1 - len(context)) + context[-(self.n - 1):]
        for hlen in range(self.n - 1, -1, -1):
            history = tuple(hist[-hlen:] if hlen > 0 else ())
            if hlen == 0 or (hlen in self.ngram_counts and self.ngram_counts[hlen].get(history, 0) > 0):
                logps = [(w, self._conditional_logprob(w, history)) for w in self.vocab]
                m = max(lp for _, lp in logps)
                exps = [(w, math.exp(lp - m)) for w, lp in logps]
                Z = sum(v for _, v in exps)
                return [(w, v / Z) for w, v in exps]
        uni = 1.0 / self.V
        return [(w, uni) for w in self.vocab]

    def predict_next(self, context: List[str], top_k: int = 5, exclude_punct: bool = True) -> List[Tuple[str, float]]:
        probs = self.next_word_probs(context)
        if exclude_punct:
            probs = [(w, p) for w, p in probs if w not in {".", ",", "!", "?", ";", ":"}]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]

if __name__ == "__main__":
    with open(r'C:\Users\shaha\OneDrive\Desktop\Data-Science-Projects\Ngram-Model\training_sentences.txt', 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]

    lm = NgramLM(n=3, alpha=0.5)
    lm.fit(corpus)

ctx = tokenize("The weather changed")

print("\n==================== N-GRAM NEXT WORD PREDICTION ====================\n")

print("🔹 Input Context:")
print(f"   {ctx}\n")

print("🔹 How It Works:")
print("   The model calculates probabilities for all possible next words.")
print("   Words with higher probability appear at the top.\n")

print("🔹 Top 3 Predicted Suggestions:")
suggestions = lm.predict_next(ctx, top_k=3)
for i, (word, prob) in enumerate(suggestions, 1):
    print(f"   {i}. {word:<12} | Probability: {prob:.4f}")

print("\n====================================================================\n")
