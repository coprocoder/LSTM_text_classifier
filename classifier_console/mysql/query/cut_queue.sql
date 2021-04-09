SET SQL_SAFE_UPDATES = 0;
UPDATE tickets_cut_queue SET ticket_queue_name = substr(ticket_queue_name,1,instr(ticket_queue_name,'::') - 1) 
	WHERE ticket_queue_name LIKE '%::%';
SELECT * FROM otrs.tickets_cut_queue;