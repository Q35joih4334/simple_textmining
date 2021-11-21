# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:44:01 2021

@author: Q35joih4334
"""

from simple_textmining import simple_textmining
import pandas as pd
import textacy.datasets

# Test data: IMDB reviews

ds = textacy.datasets.IMDB()
texts = []
for i, record in enumerate(ds.records(limit=300)):
    texts.append(record.text)
df = pd.DataFrame(texts, columns=['review'])


bt = simple_textmining(
    df, 
    'review', 
    n_topics=10)
bt.build_xlsx_report('df10.xlsx')

bt.n_topics = 20
bt.build_xlsx_report('df20.xlsx')
