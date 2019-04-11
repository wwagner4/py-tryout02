from spacy.lang.de import German

nlp = German()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
doc = nlp(u"""
14. Davon ich allzeit froehlich sei,
Zu springen, singen immer frei
Das rechte Susannine* schon,
Mit Herzen Lust den suessen Ton.

15. Lob, Ehr sei Gott im hoechsten Thron,
Der uns schenkt seinen ein'gen Sohn,
Des freuen sich der Engel Schaar
Und singen uns solch's neues Jahr.
""")
for sent in doc.sents:
    print(sent.text)
