class Token2id:
    def __init__(self):
        pass
    
    def fit(self, sentences):
        pos_tags = {}
        # positions_and_heads = {}
        dep_tags = {}
        morph_tags = {}

        # for sentence in sentences:
        #     for token_info in sentence:
        #         _, pos, _, _, dep_tag, morph = token_info
        #         if pos not in pos_tags: 
        #             pos_tags[pos] = len(pos_tags)
        #         # if position not in positions_and_heads: 
        #         #     positions_and_heads[pos] = len(positions_and_heads)
        #         if dep_tag not in dep_tags: 
        #             dep_tags[dep_tag] = len(dep_tags)
        #         for m in morph:
        #             if m not in morph_tags: 
        #                 morph_tags[pos] = len(morph_tags)

        properties = {}
        properties['NULL'] = len(properties)

        for sentence in sentences:
            for token_info in sentence:
                morph = token_info[-1]
                rest = token_info[:-1]

                for item in rest:
                    if item not in properties:
                        properties[item] = len(properties)
                
                for item in morph:
                    if item not in properties:
                        properties[item] = len(properties)

        self.properties = properties

        # collect frequencies to know which properties are the most common


        # self.pos_tags = pos_tags
        # self.dep_tags = dep_tags
        # self.morph_tags = morph_tags

    def transform(self, sentences):
        tr_sentences = []
        tr_sentence = []

        for sentence in sentences:
            for token_info in sentence:
                morph = token_info[-1]
                rest = token_info[:-1]

                prop_ids = [self.properties.get(item, 0) for item in rest]
                prop_ids.extend([self.properties.get(item, 0) for item in morph])
                tr_sentence.append(prop_ids)        
            tr_sentences.append(tr_sentence)
        
        return tr_sentences

    def inverse(self, sentence):
        prop, p_id = zip(*self.properties.items())
        inverse_dict = dict(zip(p_id, prop))

        return [[inverse_dict[p] for p in properties] for properties in sentence]

    @property
    def size(self):
        return len(self.properties)