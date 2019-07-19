class Label2id:
    def __init__(self):
        pass
    
    def fit(self, sentences):

        labels = {}
        labels['NULL'] = len(labels)

        for sentence in sentences:
            for label in sentence:
                if label not in labels:
                    labels[label] = len(labels)

        self.labels = labels

    def transform(self, sentences):
        tr_sentences = []

        for sentence in sentences:
            label_ids = [self.labels[item] for item in sentence]  
            tr_sentences.append(label_ids)
        
        return tr_sentences

    def inverse(self, sentence):
        prop, p_id = zip(*self.labels.items())
        inverse_dict = dict(zip(p_id, prop))

        return [[inverse_dict[p] for p in properties] for properties in sentence]

    @property
    def size(self):
        return len(self.labels)