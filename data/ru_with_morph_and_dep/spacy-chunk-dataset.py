import spacy
import json
import sys
from spacy.lang.ru import Russian

input_file = sys.argv[1]
model_location = sys.argv[2]

sentences = json.loads(open(input_file).read().strip())

nlp = spacy.load(model_location)

def get_mode(tag):
    parts = tag.split('-')
    if len(parts) == 2:
        mode = parts[0]
        ch_type = parts[1]
    else:
        mode = ""
        ch_type = tag
    return mode, ch_type

for sentence, labels in sentences:
    entities = labels['entities']
    doc = nlp(sentence)
    correct = spacy.gold.biluo_tags_from_offsets(doc, entities)
    
    for token, chunk in zip(doc, correct):
        tag = " ".join(token.tag_.split("|"))
        print(token.text, token.pos_, token.i, token.head.i, token.dep_, tag, chunk)
    print()


