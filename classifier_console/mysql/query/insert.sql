#SELECT * FROM vk_bot_ticket_creator.messages;
USE OTRS;
INSERT INTO tickets (TicketID, Queue, PriorityID, CustomerUserID, Title, Body, State, Type) 
VALUES (10, "Test3", 3, "ILYA", "Test", "верните данные в базу данных", "open", "Unclassified")