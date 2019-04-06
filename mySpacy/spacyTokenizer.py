import spacy
from spacy.lang.de import German
from spacy.attrs import *


txt = """
Ihr naht euch wieder, schwankende Gestalten,
Die früh sich einst dem trüben Blick gezeigt.
Versuch ich wohl, euch diesmal festzuhalten?
Fühl ich mein Herz noch jenem Wahn geneigt?
Ihr drängt euch zu! nun gut, so mögt ihr walten,
Wie ihr aus Dunst und Nebel um mich steigt;
Mein Busen fühlt sich jugendlich erschüttert
Vom Zauberhauch, der euren Zug umwittert.
"""

print(txt)

_nlp = spacy.load('de_core_news_sm')

tokenizer = German().Defaults.create_tokenizer(_nlp)

docs = tokenizer.pipe(txt)

for d in docs:
    for t in d.to_array([PROB, TAG, LENGTH, POS, ENT_TYPE, IS_ALPHA]):
        print(t)




