USE OTRS;
DROP TABLE OTRS.tickets_from_excel;
CREATE TABLE OTRS.tickets_from_excel (
		`id` MEDIUMINT UNSIGNED NOT NULL auto_increment,
		`ticket_id` MEDIUMINT UNSIGNED NOT NULL,
        `ticket_create_time` datetime NOT NULL, 
        `ticket_queue_id` TINYINT UNSIGNED UNSIGNED NOT NULL, 
        `ticket_queue_name` VARCHAR(50)  NOT NULL, 
        `ticket_service_id` SMALLINT UNSIGNED, 
        `ticket_service_name` VARCHAR(150), 
        `article_a_from` VARCHAR(120)  NOT NULL, 
        `article_a_to` TEXT  NOT NULL, 
        `article_a_subject` VARCHAR(500) NOT NULL,
        `article_a_body` TEXT NOT NULL,
        PRIMARY KEY(`id`)
        ) DEFAULT CHARSET=utf8;