# for clean text
import pymorphy2 as pm
import re
import numpy as np

# for excel r/w
import xlrd, xlwt
from pymysql import NULL

class Excel:

    @staticmethod
    def clean_text(text):
        text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
        text = text.lower()

        text = text.replace("добрый день", " ").replace("добрый вечер", " ").replace("доброе утро", " ")
        text = text.replace("день добрый", " ").replace("утро доброе", " ").replace("вечер добрый", " ")
        text = text.replace("доброго времени суток", " ").replace("здравствуйте", " ").replace("с уважением", " ")

        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
        text = " ".join(pm.MorphAnalyzer().parse(np.unicode(word))[0].normal_form for word in text.split())
        text = ' '.join(word for word in text.split() if len(word) > 3)

        return text

    @staticmethod
    def db_to_excel(self, begin_data, end_data ):
        # coding: utf-8
        from sqlalchemy import create_engine
        import pymysql
        import pandas as pd

        # print('Запускаю скрипт')
        conn = create_engine('mysql+pymysql://login:password,M1@10.1.10.5/otrs')
        # print('Соединился с базой')
        query = """SELECT
        ticket.id AS ticket_id,
        ticket.create_time as ticket_create_time,
        ticket.queue_id AS ticket_queue_id,
        ticket_queue.name AS ticket_queue_name,
        ticket.service_id AS ticket_service_id,
        service.name AS ticket_service_name,
        article.a_from AS article_a_from,
        article.a_to AS article_a_to,
        article.a_subject AS article_a_subject,
        article.a_body AS article_a_body
        FROM otrs.ticket AS ticket
        LEFT JOIN otrs.ticket_history AS ticket_history ON ticket.id = ticket_history.ticket_id
        LEFT JOIN otrs.queue AS ticket_queue ON ticket_queue.id = ticket.queue_id
        LEFT JOIN otrs.queue AS ticket_history_queue ON ticket_history_queue.id = ticket_history.queue_id
        LEFT JOIN otrs.service AS service ON service.id = ticket.service_id
        LEFT JOIN otrs.ticket_history_type AS ticket_history_type ON ticket_history_type.id = ticket_history.history_type_id
        LEFT JOIN otrs.users AS users_create_by ON users_create_by.id = ticket_history.create_by
        LEFT JOIN otrs.users AS users_change_by ON users_change_by.id = ticket_history.change_by
        LEFT JOIN otrs.users AS users_owner_id ON users_owner_id.id = ticket_history.owner_id
        INNER JOIN otrs.article AS article ON article.id = ticket_history.article_id
        LEFT JOIN otrs.ticket_state AS ticket_state ON ticket_state.id = ticket_history.state_id
        WHERE ticket.create_time BETWEEN '""" + begin_data + """ 0:00:00' AND '""" + end_data + """ 23:59:59'
        AND ticket_history.history_type_id = 12
        AND users_create_by.last_name <> 'Zabbix'
        AND ticket.queue_id <> 3
        AND ticket.ticket_state_id IN (2,3)"""

        # print('Запускаю запрос:')
        df = pd.read_sql(query, conn)
        # print('Выполнил запрос')
        df.to_excel(r'd:/tickets.xlsx', index=False)
        # print('готово')

        return df.shape[0]

    @staticmethod
    def excel_to_db(self):
        # открываем файл
        doc = xlrd.open_workbook('./tickets.xlsx')

        # выбираем активный лист
        sheet = doc.sheet_by_index(0)

        for i in range(0, sheet.nrows):
            print(i)
            if (sheet.row_values(i)[4] == ''):
                serv_id = NULL
            else:
                serv_id = round(sheet.row_values(i)[4])

            if (sheet.row_values(i)[5] == ''):
                serv_name = NULL
            else:
                serv_name = '"' + sheet.row_values(i)[5].replace('\\', '\\\\').replace('"', '\\"') + '"'

            stmt = "INSERT INTO {database}.{table} " \
                   "(ticket_id, ticket_create_time, ticket_queue_id, " \
                   "ticket_queue_name, ticket_service_id, ticket_service_name, article_a_from, article_a_to, " \
                   "article_a_subject, article_a_body) " \
                   "VALUES ({ticket_id}, {ticket_create_time}, {ticket_queue_id}, {ticket_queue_name}, " \
                   "{ticket_service_id}, {ticket_service_name}, {article_a_from}, {article_a_to}, " \
                   "{article_a_subject}, {article_a_body})".format(
                database=self.dbinfo['database'],
                table=self.dbinfo['table'],
                ticket_id=round(sheet.row_values(i)[0]),
                ticket_create_time='"' + str(
                    xlrd.xldate.xldate_as_datetime(sheet.row_values(i)[1], doc.datemode)) + '"',
                ticket_queue_id=round(sheet.row_values(i)[2]),
                ticket_queue_name='"' + sheet.row_values(i)[3].replace('\\', '\\\\').replace('"', '\\"') + '"',
                ticket_service_id=serv_id,
                ticket_service_name=serv_name,
                article_a_from='"' + sheet.row_values(i)[6].replace('\\', '\\\\').replace('"', '\\"') + '"',
                article_a_to='"' + sheet.row_values(i)[7].replace('\\', '\\\\').replace('"', '\\"') + '"',
                article_a_subject='"' + sheet.row_values(i)[8].replace('\\', '\\\\').replace('"', '\\"') + '"',
                article_a_body='"' + sheet.row_values(i)[9].replace('\\', '\\\\').replace('"', '\\"') + '"'
            )

            print('Запускаю запрос')
            con = self.get_connection()
            cur = con.cursor()
            cur.execute(stmt)
            con.commit()
            print('Выполнил запрос')



