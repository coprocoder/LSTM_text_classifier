SELECT * FROM OTRS.tickets 
	WHERE LOCATE(TRIM(LOWER("Test")), Title) > 0 or LOCATE(TRIM(LOWER("Hello")), Body) > 0;