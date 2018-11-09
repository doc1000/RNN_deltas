import psycopg2

class Connection(object):
    def __init__(self):
        self.reset()
        self.column_names = None

    def reset(self):
        self.__conn = psycopg2.connect("host=eepg1.enduranceenergy.com dbname=data user=wrfuser password=EnergyTrader2011")
        self.__cursor = self.__conn.cursor()

    def commit(self):
        return self.__conn.commit()

    def query(self, query_string, *args):
        self.__cursor.execute(query_string, args)
        self.column_names = [desc[0] for desc in self.__cursor.description]
        return self.__cursor.fetchall()

    def execute(self, query_string, *args):
        self.__cursor.execute(query_string, args)
        self.commit()

    def copy_from(self, file_obj, table_name, sep=",", columns=None):
        self.__cursor.copy_from(file_obj, table_name, sep=sep, columns=columns)
        self.commit()

    def copy_to(self, file_obj, sep=",", columns=None):
        self.__cursor.copy_to(file_obj, sep=sep, columns=columns)
