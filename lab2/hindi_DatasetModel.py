import re
from torch.utils.data import Dataset

class HindiDataset(Dataset):
    def __init__(self, datapath):
        self.sentences = []
        self.labels = []
        self.set_label_names = set()

        self._parse_file(datapath)

        self.label_names =sorted(list(self.set_label_names))
        self.label_id = {label: i for i, label in enumerate(self.label_names)}
        self.id_label = {i: label for label, i in self.label_id.items()}
        
    def _parse_file(self, datapath):
        "Parse CoNLL-U format file into sentences consisting of tokens and chunk labels"
        current_tokens = []
        current_labels = []
        
        with open(datapath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: #ckeck for empty line
                    if current_tokens:
                        self.sentences.append(current_tokens)
                        self.labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                    continue
                if line.startswith('#'):
                    continue
                if self._starts_with_digit(line):
                    parts = line.split('\t')
                    if len(parts)>= 10:
                        token = parts[1]
                        chunk_info = parts[9]

                        chunk_label = self._extract_chunkId(chunk_info)

                        if chunk_label is not None:
                            current_tokens.append(token)
                            current_labels.append(chunk_label)
                            self.set_label_names.add(chunk_label)

        if current_tokens:
            self.sentences.append(current_tokens)
            self.labels.append(current_labels)


    def _starts_with_digit(self, line):
        return bool(re.match(r'^\d', line))

    def _extract_chunkId(self, chunk_info):
        match = re.search(r'ChunkId=([^|]+)', chunk_info)
        return match.group(1) if match else None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        "Returns one sentence with its tokens and labels"
        return {'tokens': self.sentences[idx], 'labels': [self.label_id[label] for label in self.labels[idx]]}

