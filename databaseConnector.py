import jaydebeapi
import os
import ctypes


ctypes.CDLL(r'C:\Program Files\Java\jre1.8.0_191\bin')

conn = jaydebeapi.connect("org.h2.Driver",  # driver class
                          "jdbc:h2:~/facenality-server",  # JDBC url
                          ["SA", ""],  # credentials
                          "./h2.jar",)  # location of H2 jar
try:
    curs = conn.cursor()
    # Fetch the last 10 timestamps
    curs.execute("SELECT * FROM QUESTIONNAIRE")
    for value in curs.fetchall():
        # the values are returned as wrapped java.lang.Long instances
        # invoke the toString() method to print them
        print(value[0].toString())
finally:
    if curs is not None:
        curs.close()
    if conn is not None:
        conn.close()
