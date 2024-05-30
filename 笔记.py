import cv2
import numpy
import matplotlib

# 第一个参数为文件名，第二个参数：1(rgb图)，0(灰度图)，-1(包含透明通道的彩色图)
# img = cv2.imread("testimg1.png", 0)
# print(img)

# 可以先使用cv2.namedWindow()创建一个窗口，也可以直接调用cv2.imshow()显示图片
# cv2.WINDOW_NORMAL表示窗口大小可以调整，cv2.WINDOW_AUTOSIZE表示窗口大小自适应图片
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# cv2.imshow("test", img)

# 表示让程序等待, 参数为多少毫秒，0表示一直等待
# 等待期间也可以获取用户的按键输入：k = cv2.waitKey(0)
# cv2.waitKey(0)

# 使用cv2.imwrite()保存图片，参数 1 是包含后缀名的文件名，参数 2 是 img


# 使用cv2.VideoCapture(0)创建 VideoCapture 对象，参数 0 指的是摄像头的编号。
# 如果你电脑上有两个摄像头的话，访问第 2 个摄像头就可以传入 1，依此类推。
# 如果把摄像头的编号换成视频的路径就可以播放本地视频了
# 设置摄像头分辨率
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 设置窗口为全屏
cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# OpenCV人脸识别分类器
classifier1 = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classifier2 = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
# classifier3 = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')

# 定义矩形颜色
color = (0,255,0)

# capture.read()函数返回的第 1 个参数 ret(return value 缩写) 是一个布尔值，表示当前这一帧是否获取正确。第二个参数为一帧
# cv2.cvtColor()用来转换颜色，这里将彩色图转成灰度图。(低一维可以减少计算量)
# 使用cv2.rectangle()来绘制矩形，参数一为图片，参数二为左上角坐标，参数三为右下角坐标，参数四为矩形厚度
while True:
    ret, frame = capture.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 识别人脸
        faceRects = classifier1.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
        if len(faceRects):
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 人脸
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # 眼睛
                roi_gray_eye = gray[y:y + h//2, x:x + w]
                eyeRects = classifier2.detectMultiScale(roi_gray_eye, scaleFactor=1.1, minNeighbors=5)
                for eyeRect in eyeRects:
                    ex, ey, ew, eh = eyeRect
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    axesLength = (ew // 2, eh // 2)
                    cv2.ellipse(frame, center, axesLength, 0, 0, 360, color, 2)

                # 嘴巴q
                # roi_gray_mouth = gray[y + h//2:y + h, x:x + w]
                # mouthRects = classifier3.detectMultiScale(roi_gray_mouth, scaleFactor=1.01, minNeighbors=5)
                # for mouthRect in mouthRects:
                #     mx, my, mw, mh = mouthRect
                #     cv2.rectangle(frame, (x + mx, y + h//2 + my), (x + mx + mw, y + h//2 + my + mh), color, 2)
                # 嘴巴
                cv2.rectangle(frame, (x + 3 * w // 8, y + 3 * h // 4),(x + 5 * w // 8, y + 7 * h // 8), color)

        cv2.imshow("test", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()