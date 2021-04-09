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
    def get_data_from_db(self, num_tickets):
        stmt = "SELECT * FROM {database}.{table} WHERE id < {num_tickets}".format(
            database=self.dbinfo['database'],
            table=self.dbinfo['table'],
            num_tickets=int(num_tickets) + 1)

        # Read SQL data
        df = pd.read_sql(stmt, self.connection)

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
    def get_1_ticket_from_db(self, num_tickets):
        stmt = "SELECT * FROM {database}.{table} WHERE id = {num_tickets}".format(
            database=self.dbinfo['database'],
            table=self.dbinfo['table'],
            num_tickets=num_tickets)

        # Read SQL data
        ticket = pd.read_sql(stmt, self.connection)
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

    @staticmethod
    def get_ticket_from_otrs(self):

        UserLogin = self.dbotrs['username']
        Password = self.dbotrs['password']

        # Connect ------------------------------------------------------------------------------------------------
        paramsConnect = (
            ('UserLogin', UserLogin),
            ('Password', Password),
            ('DynamicFields', '0'),
            ('AllArticles', '0'),
        )

        response = requests.post(
            'http://10.1.10.5/otrs/nph-genericinterface.pl/Webservice/GenericTicketConnectorREST/Session',
            params=paramsConnect, verify=False)
        print(response.text)

        text = response.text
        text.split()
        SessionID = text[14:46]
        print(SessionID)

        # TicketSearch = Получение списка заявок ------------------------------------------------------------------
        paramsSearch = (
            ('UserLogin', UserLogin),
            ('Password', Password),
            ('State', 'open'),
            ('StateType', 'open'),
            # ('QueueID', 2),
            # ('Queue', 'TechSupport'),
            ('DynamicFields', '0'),
            ('AllArticles', '0'),

        )
        responseSearch = requests.get(
            'http://10.1.10.5/otrs/nph-genericinterface.pl/Webservice/GenericTicketConnectorREST/TicketSearch',
            params=paramsSearch)
        print(responseSearch.text)

        # TicketGet = Получение информации о заявке и фильтрация ----------------------------------------------------
        paramsGet = (
            ('UserLogin', UserLogin),
            ('Password', Password),
            ('DynamicFields', '0'),
            ('AllArticles', '1'),
        )

        list = responseSearch.json()
        print(list['TicketID'])

        k = 0
        # responseGetTicket = list['TicketID'][1]
        for ticketID in list['TicketID']:
            k += 1
            # print("ID = " + ticketID)
            # if(int(ticketID) == 623621) :
            if k == 1:
                responseGetTicket = requests.get(
                    'http://10.1.10.5/otrs/nph-genericinterface.pl/Webservice/GenericTicketConnectorREST/Ticket/' + ticketID,
                    params=paramsGet)

                # Обработка текста заявки --------------------------------------------------------------------
                TicketInfo = responseGetTicket.text
                print(type(TicketInfo))
                print(TicketInfo)
                beginText = TicketInfo.find('Body') + 7  # 803
                endText = TicketInfo.find('SLA') - 3  # 1046
                ticketText = TicketInfo[beginText:endText]  # Выделение текста заявки
                ticketText = ticketText.lower()


        headers = {
            'Content-Type': 'application/json'
        }

        paramsUpdate = (
            ('UserLogin', UserLogin),
            ('Password', Password),
            ('DynamicFields', '0'),
            ('AllArticles', '1')
        )
        responseTicketUpdate = requests.patch('http://10.1.10.5/otrs/nph-genericinterface.pl/Webservice/GenericTicketConnectorREST/TicketUpdate/' + ticketID, headers=headers, params=paramsUpdate, data=json.dumps({'Ticket':{'QueueID': QueueID}}), verify=False)
        print(responseTicketUpdate.text)