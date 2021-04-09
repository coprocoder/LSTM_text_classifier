# for clean text
import pymorphy2 as pm
import re
from numpy import unicode

# for get data from DB
import pandas as pd  # SQL
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence

# For get from OTRS
import requests # для подключения к отрс
import json

class Getter:

    @staticmethod
    def clean_text(text):
        text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
        text = text.lower()
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
        text = " ".join(pm.MorphAnalyzer().parse(unicode(word))[0].normal_form for word in text.split())
        text = ' '.join(word for word in text.split() if len(word) > 3)
        return text

    @staticmethod
    def get_data_from_db(classifier_object, num_tickets):
        stmt = "SELECT * FROM {database}.{table} WHERE id < {num_tickets}".format(
            database=classifier_object.dbinfo['database'],
            table=classifier_object.dbinfo['table'],
            num_tickets=int(num_tickets) + 1)

        # Read SQL data
        df = pd.read_sql(stmt, classifier_object.connection)

        print("Текст до обработки")
        print(df[['ticket_queue_name', 'article_a_subject', 'article_a_body']])

        # создадим массив, содержащий уникальные категории из нашего DataFrame
        categories = {}
        for key, value in enumerate(df[u'ticket_queue_name'].unique()):
            categories[value] = key + 1

        label_categories = df['ticket_queue_name'].unique()
        print("Список категорий заявок из БД для классификации: ", label_categories)

        # Запишем в новую колонку числовое обозначение категории
        df['ticket_queue_name'] = df[u'ticket_queue_name'].map(categories)

        total_categories = len(df[u'ticket_queue_name'].unique())
        print('Всего категорий: {}'.format(total_categories))
        # df = df.sample(frac=1).reset_index(drop=True)

        return df, total_categories, label_categories

    @staticmethod
    def get_1_ticket_from_db(classifier_object, num_tickets):
        stmt = "SELECT * FROM {database}.{table} WHERE id = {num_tickets}".format(
            database=classifier_object.dbinfo['database'],
            table=classifier_object.dbinfo['table'],
            num_tickets=num_tickets)

        # Read SQL data
        ticket = pd.read_sql(stmt, classifier_object.connection)
        # print("ticket for classify  = \n", ticket[['ticket_id', 'ticket_queue_name', 'article_a_body']])

        # X prepare ==============================================================
        descriptions = ticket['article_a_body']

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(descriptions.tolist())
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

        data_size = len(textSequences)
        X_train = textSequences[:data_size]

        # Максимальное количество слов в самом длинном описании заявки
        max_words = 0
        for desc in descriptions.tolist():
            words = len(desc.split())
            if words > max_words:
                max_words = words

        X_train = sequence.pad_sequences(X_train, maxlen=max_words)

        return X_train, ticket