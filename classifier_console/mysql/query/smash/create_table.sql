USE OTRS;
#DROP TABLE OTRS.tickets;
CREATE TABLE OTRS.tickets (
		`TicketID` MEDIUMINT UNSIGNED NOT NULL,
		`Queue` VARCHAR(50)  NOT NULL,
        `PriorityID` MEDIUMINT UNSIGNED  NOT NULL,
        `CustomerUserID` VARCHAR(100)  NOT NULL,
        `Title` LONGTEXT  NOT NULL,
        `Body` LONGTEXT  NOT NULL,
        `State` VARCHAR(20)  NOT NULL,
        `Type` VARCHAR(50)  NOT NULL,
        PRIMARY KEY(`TicketID`)
        ) DEFAULT CHARSET=utf8;