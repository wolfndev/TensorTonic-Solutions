import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words in sorted order.
        """
        # 1. Feste Spezial-Token IDs zuweisen
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        
        # 2. Einzigartige Wörter sammeln (sicher gegen leere Listen [])
        unique_words = {word.lower() for text in texts for word in text.split()}
        
        # 3. Wörter alphabetisch sortieren und ab ID 4 hinzufügen
        sorted_words = sorted(unique_words)
        counter = 4
        for word in sorted_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = counter
                self.id_to_word[counter] = word
                counter += 1
                
        # 4. Gesamte Vokabulargröße setzen
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK (1) for unknown words.
        """
        encoded = []
        # Text kleinschreiben und nach Whitespace trennen
        words = [word.lower() for word in text.split()]
        
        for word in words:
            if word in self.word_to_id:
                encoded.append(self.word_to_id[word])
            else:
                encoded.append(1)  # 1 = <UNK>
        return encoded
            
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Returns a space-joined string.
        Uses <UNK> for unknown IDs.
        """
        result = []
        for token_id in ids:
            # .get() liefert das Wort oder greift auf das unk_token zurück, falls die ID fehlt
            word = self.id_to_word.get(token_id, self.unk_token)
            result.append(word)

        # Mit Leerzeichen verbunden als String zurückgeben
        return " ".join(result)
