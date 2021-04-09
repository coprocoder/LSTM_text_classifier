SET SQL_SAFE_UPDATES = 0;
CREATE TABLE otrs.tickets_normalise SELECT * FROM otrs.tickets_normalise_backup;

'''CREATE TABLE `tickets_normalise` (
  `id` mediumint(8) unsigned NOT NULL DEFAULT '0',
  `ticket_id` mediumint(8) unsigned NOT NULL,
  `ticket_create_time` datetime NOT NULL,
  `ticket_queue_id` tinyint(3) unsigned NOT NULL,
  `ticket_queue_name` varchar(50) CHARACTER SET utf8 NOT NULL,
  `ticket_service_id` smallint(5) unsigned DEFAULT NULL,
  `ticket_service_name` varchar(150) CHARACTER SET utf8 DEFAULT NULL,
  `article_a_from` varchar(120) CHARACTER SET utf8 NOT NULL,
  `article_a_to` text CHARACTER SET utf8 NOT NULL,
  `article_a_subject` varchar(500) CHARACTER SET utf8 NOT NULL,
  `article_a_body` text CHARACTER SET utf8 NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
tickets_normalise'''