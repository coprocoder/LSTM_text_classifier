class Setter:

    @staticmethod
    def update_ticket_on_db(classifier_object, ticket, predicted_label):

        con = classifier_object.get_connection()

        stmt_select = "SELECT article_a_body FROM {database}.{table} Where id = {id};".format(
            id=ticket.iloc[0]['ticket_id'],
            database=classifier_object.dbinfo['database'],
            table=classifier_object.dbinfo['table'])
        cur = con.cursor()
        cur.execute(stmt_select)

        stmt_update = "UPDATE {database}.{table} SET ticket_queue_name = {new_queue} " \
                      "WHERE ticket_id = {id} AND ticket_queue_name = {old_queue}".format(
                database=classifier_object.dbinfo['database'],
                table=classifier_object.dbinfo['table'],
                new_queue = '"' + predicted_label + '"',
                id=ticket.iloc[0]['ticket_id'],
                old_queue = '"' + ticket.iloc[0]['ticket_queue_name'] + '"')

        cur.execute(stmt_update)
        con.commit()

        print('Заявка обновлена.\n')