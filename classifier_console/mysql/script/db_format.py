# for clean text
import pymorphy2 as pm
import re
import numpy as np

class DB_formalizer:

    @staticmethod
    def clean_text(text):
        text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
        text = text.lower()

        with open('stop_words.txt') as stop_words_file:
            for line in stop_words_file:
                text = text.replace(line.replace("\n","").lower(), " ")

        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
        text = " ".join(pm.MorphAnalyzer().parse(np.unicode(word))[0].normal_form for word in text.split())
        text = ' '.join(word for word in text.split() if len(word) > 3)
        return text

    @staticmethod
    def normalise_body_on_db(self, begin, end):
        con = self.get_connection()

        print('\nНачинаю форматирование.')
        for i in range(begin, end):
            print(i)
            stmt_select = "SELECT article_a_body FROM {database}.{table} Where id = {id};".format(
                id=i,
                database=self.dbinfo['database'],
                table=self.dbinfo['table'])
            cur = con.cursor()
            cur.execute(stmt_select)

            text = ""
            for j, line in enumerate(cur):
                text = line['article_a_body']
                # print("text = " + text)

            stmt_update = "UPDATE {database}.{table} SET {article_a_body} = {text} WHERE id = {id};".format(
                database=self.dbinfo['database'],
                table=self.dbinfo['table'],
                text='"' + DB_formalizer.clean_text(text) + '"',
                article_a_body=self.dbinfo['body_field'],
                id=i)
            cur.execute(stmt_update)

            # добавить SET SQL_SAFE_UPDATES = 0;  если не изменяет записи

            stmt_cut = "UPDATE {database}.tickets_test SET {ticket_queue_name} = substr({ticket_queue_name},1,instr({ticket_queue_name},'::') - 1) " \
                          "WHERE ticket_queue_name LIKE '%::%' AND id = {id};".format(
                database=self.dbinfo['database'],
                table=self.dbinfo['table'],
                id=i,
                article_a_body=self.dbinfo['body_field'],
                ticket_queue_name=self.dbinfo['queue_field'])

            cur.execute(stmt_cut)
            con.commit()

        print('Форматирование завершено.\n')




