SELECT votedby FROM hall_of_fame WHERE yearid = 2000 GROUP BY votedby ORDER BY count(*) DESC LIMIT 1

select hall_of_fame.votedby from hall_of_fame where yearid = 2000 group by votedby order by count(*) desc limit 1;