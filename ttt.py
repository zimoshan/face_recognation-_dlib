import io
import sqlite3
import zlib
import numpy as np


def loadDataBase(type):
    conn = sqlite3.connect("inspurer.db")
    cur = conn.cursor()
    if type == 1:
        knew_id = []
        knew_name = []
        knew_face_feature = []
        cur.execute('select id,name,face_feature from worker_info')
        origin = cur.fetchall()
        for row in origin:
            print(row[0])
            print(row[1])
            print(row[2])
            #knew_id.append(row[0])
            #knew_name.append(row[1])
            #knew_face_feature.append(convert_array(row[2]))

    pass

def convert_array(self,text):
        out = io.BytesIO(text)
        out.seek(0)
        dataa = out.read()
        out = io.BytesIO(zlib.decompress(dataa))
        return np.load(out)


loadDataBase(1)
