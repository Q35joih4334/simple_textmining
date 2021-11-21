# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 11:24:27 2021

@author: Q35joih4334
"""

import sys
import io
import textwrap
import spacy
import textacy.extract
import textacy.tm
import textacy.representations
import pandas as pd
import tqdm
import numpy as np
from wordcloud import WordCloud
import xlsxwriter
import matplotlib
import matplotlib.pyplot as plt
tqdm.tqdm.pandas()

def mpl_to_xlsx(worksheet, row, col, fig):

    """
    Simple function for entering matplotlib figures to xlsxwriter worksheet
    """

    imgdata = io.BytesIO()
    fig.savefig(imgdata)
    imgdata.seek(0)
    worksheet.insert_image(row, col, '', {'image_data': imgdata})

class simple_textmining:

    def __init__(self,
                 df,
                 text_column,
                 ngrams=(1, 3),
                 nlp=None,
                 keyword_algo='sgrank', #TODO: could be list
                 keyword_topn=10,
                 cvectorizer_args=None,
                 n_topics=20, # TODO: could be list?
                 model_type='nmf',
                 tvectorizer_args=None,
                 timeseries_column=None,
                 timeseries_epoch='Y',
                 docs=None):

        self.df_init = df
        self.text_column = text_column
        self.ngrams = ngrams
        self.keyword_algo = keyword_algo
        self.keyword_topn = keyword_topn
        self.n_topics = n_topics
        self.model_type = model_type
        self.timeseries_column = timeseries_column
        self.timeseries_epoch = timeseries_epoch

        # Count vectorizer args
        self.cvectorizer_args = {
            'tf_type': 'linear',
            'idf_type': None,
            'norm': None,
            'min_df': .1,
            'max_df': .95,
            'max_n_terms': 100000
            }
        if cvectorizer_args:
            self.cvectorizer_args.update(cvectorizer_args)

        # TFIDF vectorizer args
        self.tvectorizer_args = {
            'tf_type': 'linear',
            'idf_type': 'smooth',
            'norm': 'l2',
            'min_df': 3,
            'max_df': .95,
            'max_n_terms': 100000
            }
        if tvectorizer_args:
            self.tvectorizer_args.update(tvectorizer_args)

        self.nlp = nlp
        if not self.nlp:
            self.nlp = spacy.load('en_core_web_sm')

        self.docs = docs
        self.terms_list = None

    def build(self):

        # TODO: maybe make docs and terms_list into Series rather than DataFrames

        if self.docs is None:

            tqdm.tqdm.write('Creating Spacy docs.', file=sys.stderr)

            # TODO: progress_apply is not working
            #self.docs = self.df_init[self.text_column].progress_apply(self.nlp)
            d = {}
            for row, data in tqdm.tqdm(self.df_init.iterrows()):
                d[row] = self.nlp(data[self.text_column])

            #self.docs = self.df_init[self.text_column].apply(self.nlp)

            #TODO: fix parallel processing, this gives pipe error
            #docs = [d for d in tqdm.tqdm(self.nlp.pipe(self.df[self.text_column].tolist(), n_process=8))]

            self.docs = pd.Series(d, name='_doc') #TODO: or series?

        else:
            tqdm.tqdm.write('Spacy docs already calculated. Skipping.', file=sys.stderr)

        self.df = self.df_init.copy() #TODO: not sure if this is necessary

        if self.terms_list is None:

            tqdm.tqdm.write('Building bag of words.', file=sys.stderr)

            d = {}
            for row, data in tqdm.tqdm(self.docs.iteritems()):

                clean = []

                for ngram in textacy.extract.basics.ngrams(data, n=self.ngrams):

                    # Ngams are separated with underscore
                    joined_ngram = '_'.join([x.lemma_.lower() for x in ngram])

                    if len(joined_ngram) > 2:
                        clean.append(joined_ngram)

                d[row] = clean

            self.terms_list = pd.Series(d, name='_terms_list')

        else:
            tqdm.tqdm.write('Bag of words already calculated. Skipping.', file=sys.stderr)

    def keyword_extraction(self):

        if self.keyword_algo:

            tqdm.tqdm.write('Extracting keywords.', file=sys.stderr)

            d = {}
            for row, data in tqdm.tqdm(self.docs.iteritems()):

                # TODO: allow multiple algos
                if self.keyword_algo == 'sgrank':
                    keyterms = textacy.extract.keyterms.sgrank(data, topn=self.keyword_topn)

                    # TODO: this should be a bit more robust, e.g. if there is no keyterms
                    d[row] = [x[0].lower() for x in keyterms]

            # TODO: this could be dataframe with keyword algo in header
            self.df['_top_keywords_{}'.format(self.keyword_algo)] = pd.Series(d)

    def word_counts(self):

        tqdm.tqdm.write('Running word counts.', file=sys.stderr)

        cvectorizer = textacy.representations.Vectorizer(**self.cvectorizer_args)

        count_doc_term_matrix = cvectorizer.fit_transform(self.terms_list.values)

        df_vectorized = pd.DataFrame(count_doc_term_matrix.toarray(), index=self.df.index)
        df_vectorized = df_vectorized.rename(cvectorizer.id_to_term, axis='columns')

        # Sort columns by most prevalent
        df_vectorized = df_vectorized[df_vectorized.sum().sort_values(ascending=False).index]

        self.df_vectorized = df_vectorized

    def topic_modelling(self):

        tqdm.tqdm.write('Running topic model.', file=sys.stderr)

        # NOTE: lda gives strange results with the default settings
        # TODO: include something to help choose the number of topics for topic modelling

        tvectorizer = textacy.representations.Vectorizer(**self.tvectorizer_args)

        doc_term_matrix = tvectorizer.fit_transform(self.terms_list.values)

        # Run topic model
        model = textacy.tm.TopicModel(self.model_type, n_topics=self.n_topics)
        model.fit(doc_term_matrix)
        doc_topic_matrix = model.transform(doc_term_matrix)

        # Build top terms
        top_terms_str = []
        for topic_idx, top_terms in model.top_topic_terms(tvectorizer.id_to_term):
            top_terms_str.append('TOPIC {}: {}'.format(str(topic_idx).zfill(2), ', '.join(top_terms)))

        docs_terms_weights = list(model.top_topic_terms(tvectorizer.id_to_term, weights=True, top_n=-1))

        # Get dominant topics

        # NOTE: this finds multiple dominant topics if there are
        dominant_topics = []
        for row in doc_topic_matrix:
            max_index = row.argmax()
            max_indexes = np.where(row == row[max_index])[0]
            dominant_topics.append([top_terms_str[x] for x in max_indexes])
        self.dominant_topics = pd.Series(dominant_topics, index=self.df.index)
        
        # TODO: dont do this, join when reporting tm
        self.df['_dominant_topics'] = pd.Series(dominant_topics, index=self.df.index)

        # This gets just one dominant topic
        dominant_topic = []
        for row in doc_topic_matrix:
            max_index = row.argmax()
            dominant_topic.append(top_terms_str[max_index])
        self.dominant_topic = pd.Series(dominant_topic, index=self.df.index)    
        
        # TODO: dont do this, join when reporting tm
        self.df['_dominant_topic'] = pd.Series(dominant_topic, index=self.df.index)

        self.top_terms = pd.DataFrame(
            doc_topic_matrix,
            columns=top_terms_str,
            index=self.df.index)
        
        # TODO: dont do this, join when reporting tm
        self.df = self.df.join(pd.DataFrame(doc_topic_matrix,
                                            columns=top_terms_str,
                                            index=self.df.index))

        self.model = model
        self.doc_term_matrix = doc_term_matrix
        self.top_terms_str = top_terms_str
        self.doc_topic_matrix = doc_topic_matrix
        self.tvectorizer = tvectorizer
        self.docs_terms_weights = docs_terms_weights

    def report_counts(self):

        table = self.df_init.join(self.df_vectorized)
        table.to_excel(self.writer, sheet_name='counts')

        worksheet = self.writer.sheets['counts']
        worksheet.freeze_panes(1, 0)

        # Add table
        columns = ['index'] + table.columns.tolist()
        columns_data = []
        for column in columns:
            columns_data.append(
                {'header': column})

        table_range = xlsxwriter.utility.xl_range(
            0,
            0,
            len(self.df.index),
            len(self.df_vectorized.columns) + len(self.df_init.columns))

        table_style = self.table_style
        table_style.update({'columns': columns_data})

        worksheet.add_table(
            table_range,
            table_style)        
        
        # Add conditional format for counts
        worksheet.conditional_format(
            1,
            len(self.df_init.columns) + 1,
            len(self.df_vectorized.index),
            len(self.df_vectorized.columns) + len(self.df_init.columns) + 1,
            {'type': '2_color_scale',
             'min_value': 0,
             'min_color': '#FFFFFF',
             'max_value': self.df_vectorized.max().max(),
             'max_color': '#4f81bd'})        

    def report_tm(self):

        tqdm.tqdm.write('Reporting topic model.', file=sys.stderr)

        table = self.df

        table.to_excel(
            self.writer,
            startrow=2,
            sheet_name='topic_model')
        worksheet = self.writer.sheets['topic_model']

        # Add table
        columns = ['index'] + table.columns.tolist()
        columns_data = []
        for column in columns:
            columns_data.append(
                {'header': column,
                 'header_format': self.hidden_format})

        table_range = xlsxwriter.utility.xl_range(
            2,
            0,
            len(table.index) + 2,
            len(table.columns))

        table_style = self.table_style
        table_style.update({'columns': columns_data})

        worksheet.add_table(
            table_range,
            table_style)

        # Top header
        for i, column in enumerate(columns):

            if column in self.top_terms_str:

                worksheet.write(0, i, column, self.topic_format)

                formula = '=COUNTIF({},"*"&{}&"*")'.format(
                    xlsxwriter.utility.xl_range(
                        3,
                        columns.index('_dominant_topics'),
                        len(table.index) + 2,
                        columns.index('_dominant_topics')),
                    xlsxwriter.utility.xl_rowcol_to_cell(0, i))
                worksheet.write_formula(1, i, formula)

            else:

                worksheet.write(0, i, column, self.header_format)

        worksheet.set_row(0, 160)

        # Format topic weights
        weights_range = xlsxwriter.utility.xl_range(
            3,
            columns.index(self.top_terms_str[0]),
            len(table.index) + 2,
            columns.index(self.top_terms_str[-1]))        
        
        worksheet.conditional_format(weights_range,
                                     {'type': '2_color_scale',
                                      'min_value': 0,
                                      'min_color': '#FFFFFF',
                                      'max_value': table[self.top_terms_str].max().max(),
                                      'max_color': '#4f6228'})

        # Hide zero weights
        worksheet.conditional_format(weights_range,
                                     {'type': 'cell',
                                      'criteria': 'equal to',
                                      'value': 0,
                                      'format': self.hidden_format})

        # Highlight dominant topic
        formula = '=ISNUMBER(SEARCH({},{}))'.format(
            xlsxwriter.utility.xl_rowcol_to_cell(2,
                                                 columns.index(self.top_terms_str[0]),
                                                 row_abs=True),
            xlsxwriter.utility.xl_rowcol_to_cell(3,
                                                 columns.index(self.top_terms_str[0]) - 1,
                                                 col_abs=True))

        worksheet.conditional_format(weights_range,
                                     {'type': 'formula',
                                      'criteria': formula,
                                      'format': self.highlighted_format})

        # Freeze top rows
        worksheet.freeze_panes(3, 0)

    def report_wordclouds(self):

        tqdm.tqdm.write('Drawing topic model wordclouds.', file=sys.stderr)

        worksheet = self.writer.book.add_worksheet('topic_wordclouds')

        for i, doc_terms_weights in enumerate(self.docs_terms_weights):

            wc_freqs = {x[0]: x[1] for x in doc_terms_weights[1]}

            if all([x == 0 for x in wc_freqs.values()]):
                continue

            wc = WordCloud(
                background_color='white',
                max_words=1000,
                scale=8,
                color_func=lambda *args, **kwargs: 'black'
                )
            wc.generate_from_frequencies(wc_freqs)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.title(textwrap.fill(self.top_terms_str[i], width=40))
            plt.tight_layout()
            mpl_to_xlsx(worksheet, i * 25, 0, fig)
            plt.close()

    def report_dominant_topics(self):

        tqdm.tqdm.write('Drawing dominant topics chart.', file=sys.stderr)

        # Counts of dominant topics
        worksheet = self.writer.book.add_worksheet('dominant_topics')

        fig, ax = plt.subplots(figsize=(16, 9))

        self.df._dominant_topics.value_counts().plot.barh(ax=ax)

        plt.tight_layout()
        mpl_to_xlsx(worksheet, 0, 0, fig)
        plt.close()

    def report_termite_plot(self):

        tqdm.tqdm.write('Drawing termite plot.', file=sys.stderr)

        # Visualise topics with termite plot
        # NOTE: n_terms should be such that all top10 terms are visible
        # TODO: highlight dominant term?
        worksheet = self.writer.book.add_worksheet('termite')

        ax = self.model.termite_plot(
            self.doc_term_matrix,
            self.tvectorizer.id_to_term,
            topics=-1,
            #n_terms=len(set(itertools.chain.from_iterable(top_terms_list))),
            sort_terms_by='seriation')
        mpl_to_xlsx(worksheet, 0, 0, ax.get_figure())
        plt.close()

    def report_timeline_chart(self):

        # TODO: maybe there should be option to set xticklabels format manually

        if self.timeseries_column:

            tqdm.tqdm.write('Drawing timeline chart.', file=sys.stderr)

            # Dominant topics
            worksheet = self.writer.book.add_worksheet('timeline_dominant_topics')

            fig, ax = plt.subplots(figsize=(16, 9))

            data = pd.crosstab(
                self.df[self.timeseries_column],
                self.df._dominant_topic)

            data = data.resample(self.timeseries_epoch).sum()
            data = data.transform(lambda x: x / x.sum(), axis=1)

            data.plot(
                ax=ax,
                kind='bar',
                width=1,
                stacked=True).legend(
                    loc='lower center',
                    bbox_to_anchor=(.5, -.5))

            ax.set_xticklabels(data.index.strftime('%' + self.timeseries_epoch))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

            plt.tight_layout()
            mpl_to_xlsx(worksheet, 0, 0, ax.get_figure())
            plt.close()

            # Count of non-zero topics
            worksheet = self.writer.book.add_worksheet('timeline_nonzero_topics')

            fig, ax = plt.subplots(figsize=(16, 9))

            data = pd.DataFrame(
                index=self.df[self.timeseries_column],
                data=(self.doc_topic_matrix != 0),
                columns=self.top_terms_str).groupby(self.timeseries_column).sum()

            data = data.resample(self.timeseries_epoch).sum()
            data = data.transform(lambda x: x / x.sum(), axis=1)

            data.plot(
                ax=ax,
                kind='bar',
                width=1,
                stacked=True).legend(
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.5))

            ax.set_xticklabels(data.index.strftime('%' + self.timeseries_epoch))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

            plt.tight_layout()
            mpl_to_xlsx(worksheet, 0, 0, ax.get_figure())
            plt.close()

    def cooccurrence_network(self):

        # NOTE: this just generates the graph but does not visualize it in any way

        tqdm.tqdm.write('Creating co-occurrence network.', file=sys.stderr)

        # TODO: this could also be calculated elsewhere
        doc_sents = []
        for doc in self.docs:
            for sent in doc.sents:
                sent_data = []
                for token in sent:
                    if not token.is_punct and not token.is_stop:
                        sent_data.append(token.lemma_.lower())
                if sent_data:
                    doc_sents.append(sent_data)

        self.G_cooccurrence = textacy.representations.network.build_cooccurrence_network(doc_sents)
        self.doc_sents = doc_sents

    def sentiment_analysis(self):

        tqdm.tqdm.write('Running sentiment analysis.', file=sys.stderr)

        # Depeche Mood
        import textacy.resources
        rs = textacy.resources.DepecheMood(lang="en", word_rep='lemmapos')
        rs.download()
        moods = {}
        for row, doc in tqdm.tqdm(self.docs.iteritems()):
            moods[row] = rs.get_emotional_valence(doc)
        self.moods = pd.DataFrame.from_dict(moods, orient='index')

        # NLTK
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        pols = {}
        for row, doc in tqdm.tqdm(self.docs.iteritems()):
            pols[row] = sia.polarity_scores(doc.text)
        self.polarities = pd.DataFrame.from_dict(pols, orient='index')

    def report_sentiment_analysis(self):

        tqdm.tqdm.write('Reporting sentiment analysis.', file=sys.stderr)

        table = self.df_init.join(self.moods.join(self.polarities))
        table.to_excel(
            self.writer,
            startrow=0,
            sheet_name='sentiment_analysis')

        worksheet = self.writer.sheets['sentiment_analysis']
        worksheet.freeze_panes(1, 0)

        columns = ['index'] + table.columns.tolist()
        columns_data = []
        for column in columns:
            columns_data.append(
                {'header': column})

        table_range = xlsxwriter.utility.xl_range(
            0,
            0,
            len(table.index),
            len(table.columns))

        table_style = self.table_style
        table_style.update({'columns': columns_data})

        worksheet.add_table(
            table_range,
            table_style)

    def report_wordcloud(self):

        tqdm.tqdm.write('Drawing wordcloud.', file=sys.stderr)

        worksheet = self.writer.book.add_worksheet('wordcloud')

        all_terms = self.terms_list.sum()
        s_all_terms = pd.Series(all_terms)
        wc_freqs = s_all_terms.value_counts().to_dict()

        wc = WordCloud(
            background_color='white',
            max_words=10000,
            scale=16,
            color_func=lambda *args, **kwargs: 'black'
            )
        wc.generate_from_frequencies(wc_freqs)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.title('Full wordcloud')
        plt.tight_layout()
        mpl_to_xlsx(worksheet, 0, 0, fig)
        plt.close()

    def report_settings(self):

        worksheet = self.writer.book.add_worksheet('settings')
        #TODO

    def define_xlsx_styles(self):

        self.topic_format = self.writer.book.add_format({
            'text_wrap': True,
            'valign': 'bottom',
            'align': 'left',
            'fg_color': '#D7E4BC',
            'rotation': 30,
            'font_size': 8,
            'border': 1})

        self.header_format = self.writer.book.add_format({
            'text_wrap': True,
            'valign': 'bottom',
            'align': 'left',
            'rotation': 30,
            'font_size': 12,
            'border': 1})

        self.hidden_format = self.writer.book.add_format({
            'font_color': '#FFFFFF'})

        self.centered = self.writer.book.add_format({
            'align': 'center'})

        self.highlighted_format = self.writer.book.add_format({
            'bold': True})
        
        # TODO: use this in tables
        self.table_style = {
            'style': 'Table Style Light 15',
            'banded_rows': False}

    def build_xlsx_report(self,
                          outfile='df.xlsx'):

        self.build()
        self.keyword_extraction()
        self.word_counts()
        self.topic_modelling()
        self.sentiment_analysis()
        self.cooccurrence_network()

        self.writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
        self.define_xlsx_styles()

        self.report_counts()
        self.report_tm()
        self.report_wordclouds()
        self.report_dominant_topics()
        self.report_termite_plot()
        self.report_timeline_chart()
        self.report_sentiment_analysis()
        self.report_wordcloud()
        self.report_settings()

        self.writer.save()
