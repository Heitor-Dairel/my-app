select CC.*, DD.ID_NOME, DD.NOMES from
(select LEVEL as id, 123 as teste, 1233 as teste1
from dual
WHERE 1 = 1--:teste
CONNECT BY LEVEL <= 4) cc LEFT join
(select LEVEL AS ID,
        CASE WHEN MOD(LEVEL,3) = 0 THEN 3 ELSE  MOD(LEVEL,3) END AS ID_NOME,
        CASE WHEN MOD(LEVEL,3) = 0 THEN 'HEITOR' ELSE  'OUTROS' END AS NOMES 
        FROM DUAL
connect BY LEVEL <=4) dd on dd.ID_NOME = cc.ID

where 1 = :teste