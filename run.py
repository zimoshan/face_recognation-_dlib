from PIL import Image, ImageDraw, ImageFont
from skimage import io as iio
import numpy as np
import threading
import imutils
import _thread
import wx.grid
import sqlite3
import cv2
import zlib
import dlib
import wx
import os
import io


# ----------------------------------------------------------------------------------------------------------------------

ID_WORKER_UNAVIABLE = -1
ID_START_PUNCHCARD = 190
ID_NEW_REGISTER = 160


# ----------------------------------------------------------------------------------------------------------------------


facerec = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
PATH_FACE = "./data/"


# ----------------------------------------------------------------------------------------------------------------------
# 计算两个特征之间的欧式距离
# ----------------------------------------------------------------------------------------------------------------------

def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))

    if dist > 0.4:
        return "diff"
    else:
        return "same"


# ---------------------------------------------------------------------------------------------
# opencv不能输出汉字，要使用Image转换一下
# cv2ImgAddText() --- 第一个参数是输出汉字的帧，或者图片，第二个参数是输出的汉字，第三个参数是输出汉字左上角位置的横坐标
# 第四个参数是输出汉字左上角位置的纵坐标，第五个参数是输出文字的颜色，第六个参数是输出文字的大小
# isinstance() 函数来判断一个对象是否是一个已知的类型 --- 第一个是要对比的对象，第二个是参照对象，一致返回True，否则返回False
# cv2.cvtColor(img, cv2.COLOR_BGR2RGB) --- 将BGR通道的图片转成灰度图
# Image.fromarray --- 实现array到image的转换
# ImageDraw.Draw(img) --- 绘制图像
# ImageFont.truetype --- 对输出文本的设置，第一个参数是字体的样式，第二个参数是是字体的大小，题三个参数是编码的格式
# draw.text --- 输出文本信息 --- 第一个参数是文本框的左上角坐标，第二个参数是文本，第三个参数是文本的颜色，第四个参数是字体设置
# return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR) --- 将灰度图转为opencv使用的BGR图
# ---------------------------------------------------------------------------------------------
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("./font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# ----------------------------------------------------------------------------------------------------------------------


class WAS(wx.Frame):

    # ----------------------------------------------------------------------------------------------------------------------
    # 初始化，创建窗口
    # ----------------------------------------------------------------------------------------------------------------------

    def __init__(self):
        wx.Frame.__init__(self, parent=None, title="人脸识别程序", size=(600, 420))
        self.initMenu()
        self.initGallery()
        self.initDatabase()
        self.initData()

    # ----------------------------------------------------------------------------------------------------------------------
    # 初始化参数
    # ----------------------------------------------------------------------------------------------------------------------

    def initData(self):
        self.name = ""
        self.id = ID_WORKER_UNAVIABLE
        self.face_feature = ""
        self.pic_num = 0
        self.flag_registed = False
        self.loadDataBase(1)

    # ----------------------------------------------------------------------------------------------------------------------
    # 初始化界面
    # ----------------------------------------------------------------------------------------------------------------------

    def initMenu(self):
        menuBar = wx.MenuBar()

        registerMenu = wx.Menu()
        self.new_register = wx.MenuItem(registerMenu, ID_NEW_REGISTER, "新建录入")
        self.new_register.SetBitmap(wx.Bitmap("drawable/new_register.png"))
        registerMenu.Append(self.new_register)

        puncardMenu = wx.Menu()
        self.start_punchcard = wx.MenuItem(puncardMenu, ID_START_PUNCHCARD, "开始识别")
        self.start_punchcard.SetBitmap(wx.Bitmap("drawable/start_punchcard.png"))
        puncardMenu.Append(self.start_punchcard)

        menuBar.Append(registerMenu, "&人脸录入")
        menuBar.Append(puncardMenu,  "&人脸识别")
        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.OnNewRegisterClicked,    id=ID_NEW_REGISTER)
        self.Bind(wx.EVT_MENU, self.OnRecognitionClicked,    id=ID_START_PUNCHCARD)

    # ----------------------------------------------------------------------------------------------------------------------
    # 点击'新建录入'调用的函数
    # ----------------------------------------------------------------------------------------------------------------------

    def register_cap(self, event):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            im_rd = imutils.resize(im_rd, width=600)
            dets = detector(im_rd, 1)

            if len(dets) != 0:
                biggest_face = dets[0]
                maxArea = 0
                for det in dets:
                    w = det.right() - det.left()
                    h = det.top()-det.bottom()
                    if w*h > maxArea:
                        biggest_face = det
                        maxArea = w*h

                cv2.rectangle(im_rd,
                              tuple([biggest_face.left(), biggest_face.top()]),
                              tuple([biggest_face.right(), biggest_face.bottom()]),
                              (255, 0, 0),
                              2)

                img_height, img_width = im_rd.shape[:2]
                image1 = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                pic = wx.Bitmap.FromBuffer(img_width, img_height, image1)
                self.bmp.SetBitmap(pic)

                shape = predictor(im_rd, biggest_face)
                features_cap = facerec.compute_face_descriptor(im_rd, shape)

                for i, knew_face_feature in enumerate(self.knew_face_feature):
                    compare = return_euclidean_distance(features_cap, knew_face_feature)
                    if compare == "same":
                        print("-------目标人脸已完成录入，请不要重复录入--------")
                        self.OnFinishRegister()
                        _thread.exit()

                face_height = biggest_face.bottom()-biggest_face.top()
                face_width = biggest_face.right()- biggest_face.left()
                im_blank = np.zeros((face_height, face_width, 3), np.uint8)

                for ii in range(face_height):
                    for jj in range(face_width):
                        im_blank[ii][jj] = im_rd[biggest_face.top() + ii][biggest_face.left() + jj]

                if len(self.name) > 0:
                    cv2.imencode('.jpg', im_blank)[1].tofile(
                    PATH_FACE + self.name + "/img_face_" + str(self.pic_num) + ".jpg")
                    self.pic_num += 1
                    print("写入本地：", str(PATH_FACE + self.name) + "/img_face_" + str(self.pic_num) + ".jpg")
                    print("图片:" + str(PATH_FACE + self.name) + "/img_face_" + str(self.pic_num) + ".jpg保存成功\r\n")

                if self.new_register.IsEnabled():
                    _thread.exit()

                if self.pic_num == 10:
                    self.OnFinishRegister()
                    _thread.exit()

    # ----------------------------------------------------------------------------------------------------------------------

    def OnNewRegisterClicked(self, event):
        self.OnNewRegister(self)

    # ----------------------------------------------------------------------------------------------------------------------

    #输入姓名、学号
    def OnNewRegister(self, event):
        self.new_register.Enable(False)
        self.loadDataBase(1)
        while self.id == ID_WORKER_UNAVIABLE:
            self.id = wx.GetNumberFromUser(message="请输入您的学号(-1不可用)",
                                           prompt="输入：",
                                           caption="温馨提示",
                                           value=ID_WORKER_UNAVIABLE,
                                           parent=self.bmp,
                                           max=1000000000,
                                           min=ID_WORKER_UNAVIABLE)
            for knew_id in self.knew_id:
                if knew_id == self.id:
                    self.id = ID_WORKER_UNAVIABLE
                    wx.MessageBox(message="学号已存在，请重新输入",
                                  caption="警告")

        while self.name == '':
            self.name = wx.GetTextFromUser(message="请输入您的的姓名,用于创建姓名文件夹", caption="温馨提示", default_value="", parent=self.bmp)

            for exist_name in (os.listdir(PATH_FACE)):
                if self.name == exist_name:
                    wx.MessageBox(message="姓名文件夹已存在，请重新输入", caption="警告")
                    self.name = ''
                    break

        os.makedirs(PATH_FACE+self.name)
        _thread.start_new_thread(self.register_cap, (event,))
        pass

    # ----------------------------------------------------------------------------------------------------------------------

    def OnFinishRegister(self):
        self.new_register.Enable(True)
        self.cap.release()

        self.bmp.SetBitmap(wx.Bitmap(self.pic_index_1))

        if self.flag_registed:
            dir = PATH_FACE + self.name

            for file in os.listdir(dir):
                os.remove(dir+"/"+file)
                print("已删除已录入人脸的图片", dir+"/"+file)

            os.rmdir(PATH_FACE + self.name)
            print("已删除已录入人脸的姓名文件夹", dir)
            self.initData()
            return

        if self.pic_num > 0:
            pics = os.listdir(PATH_FACE + self.name)
            feature_list = []
            feature_average = []

            for i in range(len(pics)):
                pic_path = PATH_FACE + self.name + "/" + pics[i]
                print("正在读的人脸图像：", pic_path)
                img = iio.imread(pic_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dets = detector(img_gray, 1)

                if len(dets) != 0:
                    shape = predictor(img_gray, dets[0])
                    face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
                    feature_list.append(face_descriptor)
                else:
                    print("未在照片中识别到人脸")

            if len(feature_list) > 0:

                for j in range(128):
                    feature_average.append(0)
                    for i in range(len(feature_list)):
                        feature_average[j] += feature_list[i][j]
                    feature_average[j] = (feature_average[j]) / len(feature_list)
                self.insertARow([self.id, self.name, feature_average], 1)
                print("学号:" + str(self.id) + "姓名:" + self.name+" 的人脸数据已成功存入\r\n")
            pass

        else:
            os.rmdir(PATH_FACE + self.name)
            print("已删除空文件夹", PATH_FACE + self.name)
        self.initData()

    # ----------------------------------------------------------------------------------------------------------------------

    #获取视频流
    def punchcard_cap(self, event):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            cv2.waitKey(1)
            im_rd = imutils.resize(im_rd, width=600)
            dets = detector(im_rd, 1)

            if len(dets) != 0:
                biggest_face = dets[0]
                maxArea = 0
                for det in dets:
                    w = det.right() - det.left()
                    h = det.top() - det.bottom()
                    if w * h > maxArea:
                        biggest_face = det
                        maxArea = w * h

                #人脸探测标记
                cv2.rectangle(im_rd,
                              tuple([biggest_face.left(), biggest_face.top()]),
                              tuple([biggest_face.right(), biggest_face.bottom()]),
                              (255, 0, 255),
                              2)

                shape = predictor(im_rd, biggest_face)
                features_cap = facerec.compute_face_descriptor(im_rd, shape)

                for i, knew_face_feature in enumerate(self.knew_face_feature):
                    compare = return_euclidean_distance(features_cap, knew_face_feature)
                    if compare == "same":

                        #打上人脸标签
                        im_rd = cv2ImgAddText(im_rd,
                                              self.knew_name[i],
                                              biggest_face.left(),
                                              biggest_face.top() - 30,
                                              (255, 0, 0),
                                              30)

                    if compare == "diff":

                        im_rd = cv2ImgAddText(im_rd,
                                              "Unknown",
                                              biggest_face.left(),
                                              biggest_face.top() - 30,
                                              (255, 0, 0),
                                              30)
                    img_height, img_width = im_rd.shape[:2]
                    image1 = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                    pic = wx.Bitmap.FromBuffer(img_width, img_height, image1)
                    self.bmp.SetBitmap(pic)

            if self.start_punchcard.IsEnabled():
                self.bmp.SetBitmap(wx.Bitmap(self.pic_index_1))
                _thread.exit()

    # ----------------------------------------------------------------------------------------------------------------------

    def OnRecognitionClicked(self, event):
        self.start_punchcard.Enable(False)
        self.loadDataBase(1)
        threading.Thread(target=self.punchcard_cap,args=(event,)).start()
        pass

    # ----------------------------------------------------------------------------------------------------------------------

    def initGallery(self):
        self.pic_index_1 = wx.Image("drawable/index.png", wx.BITMAP_TYPE_ANY).Scale(600, 500)
        self.bmp = wx.StaticBitmap(parent=self, pos=(0, 0), bitmap=wx.Bitmap(self.pic_index_1))
        self.txt = wx.StaticText(self, -1, pos=(80, 350), label=" ", size=(20, 100))
        self.txt.SetFont(wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.NORMAL, False))
        pass

    # ----------------------------------------------------------------------------------------------------------------------

    def initDatabase(self):
        conn = sqlite3.connect("inspurer.db")
        cur = conn.cursor()
        cur.execute('''create table if not exists worker_info(name text not null, id int not null primary key, face_feature array not null)''')
        cur.execute('''create table if not exists logcat(datetime text not null, id int not null, name text not null, late text not null)''')
        cur.close()
        conn.commit()
        conn.close()

    # ----------------------------------------------------------------------------------------------------------------------

    def adapt_array(self,arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        dataa = out.read()
        return sqlite3.Binary(zlib.compress(dataa, zlib.Z_BEST_COMPRESSION))

    # ----------------------------------------------------------------------------------------------------------------------

    def insertARow(self,Row,type):
        conn = sqlite3.connect("inspurer.db")
        cur = conn.cursor()
        if type == 1:
            cur.execute("insert into worker_info (id,name,face_feature) values(?,?,?)", (Row[0],Row[1],self.adapt_array(Row[2])))
            print("写人脸数据成功")

        cur.close()
        conn.commit()
        conn.close()
        pass

    # ----------------------------------------------------------------------------------------------------------------------

    def convert_array(self,text):
        out = io.BytesIO(text)
        out.seek(0)
        dataa = out.read()
        out = io.BytesIO(zlib.decompress(dataa))
        return np.load(out)

    # ----------------------------------------------------------------------------------------------------------------------

    def loadDataBase(self, type):
        conn = sqlite3.connect("inspurer.db")
        cur = conn.cursor()
        if type == 1:
            self.knew_id = []
            self.knew_name = []
            self.knew_face_feature = []
            cur.execute('select id,name,face_feature from worker_info')
            origin = cur.fetchall()
            for row in origin:
                self.knew_id.append(row[0])
                self.knew_name.append(row[1])
                self.knew_face_feature.append(self.convert_array(row[2]))

        pass


# ----------------------------------------------------------------------------------------------------------------------

app = wx.App()
frame = WAS()
frame.Show()
app.MainLoop()

# ----------------------------------------------------------------------------------------------------------------------
