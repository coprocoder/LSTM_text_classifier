class Setter:

    @staticmethod
    def update_ticket_on_db(self, ticket, predicted_label):
        con = self.get_connection()

        stmt_update = "UPDATE {database}.{table} SET ticket_queue_name = {new_queue} " \
                      "WHERE id = {id} AND ticket_queue_name = {old_queue}".format(
                database=self.dbinfo['database'],
                table=self.dbinfo['table'],
                new_queue = predicted_label,
                id=ticket.iloc[0]['ticket_id'],
                old_queue = ticket.iloc[0]['ticket_queue_name'])

        cur = con.cursor()
        cur.execute(stmt_update)
        con.commit()

        print('Заявка обновлена.\n')