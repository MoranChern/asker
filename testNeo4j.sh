# !/bin/bash

# 运行此脚本，如果输出OK，说明neo4j服务正常运行，且可以连接。

python - <<'PY'
from neo4j import GraphDatabase
uri="bolt://localhost:58287"
user="neo4j"
pwd="1kcsy2C7Vrn9JHuh"
driver=GraphDatabase.driver(uri, auth=(user,pwd))
driver.verify_connectivity()
print("OK")
driver.close()
PY
