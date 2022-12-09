import cv2
import numpy as np

# 1.讀取圖片
def 讀取圖片(name, img):
    path_image= img #圖片路徑
    image = cv2.imread(path_image) #讀取圖片
    print(image.shape)#印出圖片資訊：印出的(h, w, c)，分別代表圖片的高度、寬度與通道數(彩色為3通道)
    cv2.imshow(name,image) #顯示圖片
    cv2.waitKey(0) #顯示等待時間
def 存取圖片(img_save,img):
    image_path = img
    image = cv2.imread(image_path)
    cv2.imwrite(img_save, image)
def 讀取影片(video_path,frame_rate):# frame_rate代表每幾幀擷取一次
    cap = cv2.VideoCapture(video_path)
    # frame_rate代表每幾幀擷取一次
    # frame_rate = 3
    count = 1
    # 任意鍵換下一幀
    while (True):
        ret, frame = cap.read()
        if ret:
            if count % frame_rate == 0:
                print("擷取影片第：" + str(count) + " 幀")
                # 將擷取圖片縮小，便於顯示
                resize_img = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)
                print(resize_img.shape)#印出圖片資訊：印出的(h, w, c)，分別代表圖片的高度、寬度與通道數(彩色為3通道)
                cv2.imshow('frame', resize_img)
                cv2.waitKey(0)
            count += 1
        else:
            pass
    cap.release()
    cv2.destroyAllWindows()
    print('程式執行結束')
def 水平翻轉(name,image_path):
    # 顯示圖檔
    image = cv2.imread(image_path)
    # 水平翻轉
    image1 = cv2.flip(image, 1)
    cv2.imshow(name, image1)
    cv2.waitKey(0)

def 垂直翻轉(name,image_path):
    # 顯示圖檔
    image = cv2.imread(image_path)
    # 垂直翻轉
    image1 = cv2.flip(image, 0)
    cv2.imshow(name, image1)
    cv2.waitKey(0)
def 影像旋轉_90度旋轉(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))

    # 90度旋轉 參數：n=1代表逆時針旋轉90度、n=-1代表順時針旋轉90度
    image1 = np.rot90(image, 0)
    show_img('no rotation', image1)
    image2 = np.rot90(image, 1)
    show_img('counterclockwise90', image2)
    image3 = np.rot90(image, -1)
    show_img('clockwise90', image3)
    cv2.waitKey(0)
def 任意角度旋轉(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    show_img('origin', image)

    (h, w, c) = image.shape
    center = (w // 2, h // 2)
    # 代表逆時針旋轉30度，縮放倍數為1倍
    M = cv2.getRotationMatrix2D(center, 30, 1)
    # (w, h )代表圖片縮放與旋轉後，需裁切成的尺寸
    image1 = cv2.warpAffine(image, M, (w, h))
    show_img('counterclockwise30', image1)
def 影像放大縮小(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path)
    show_img('origin', image)
    image1 = cv2.resize(image, (250, 250))
    show_img('resize', image1)
def 比例縮放(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path)
    show_img('origin', image)
    image1 = cv2.resize(image, dsize=None, fx=0.5, fy=0.3)
    show_img('resize', image1)
def 影像分割(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path)
    show_img('origin', image)
    # 起點座標
    start_point = (50, 100)
    # 剪裁區域的寬與高
    (w, h) = (250, 300)
    image1 = image[start_point[0]:start_point[0] + w, start_point[1]:start_point[1] + h]
    show_img('crop', image1)
def 影像合併_圖像大小一致(image1_path,image2_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image1 = cv2.imread(image1_path)
    image1 = cv2.resize(image1, (250, 250))
    image2 = cv2.imread(image2_path)
    image2 = cv2.resize(image2, (250, 250))
    show_img('image1', image1)
    show_img('image2', image2)

    # 垂直合併
    vertical_image = cv2.vconcat([image1, image2])
    show_img('vertical_image', vertical_image)

    # 水平合併
    level_image = cv2.hconcat([image1, image2])
    show_img('level_image', level_image)

def 影像合併_圖像大小不一致(image1_path,image2_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image1 = cv2.imread(image1_path)
    image1 = cv2.resize(image1, (220, 220))
    image2 = cv2.imread(image2_path)
    image2 = cv2.resize(image2, (130, 130))
    show_img('image1', image1)
    show_img('image2', image2)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 水平合併
    image3 = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    image3[:h1, :w1, :3] = image1
    image3[:h2, w1:w1 + w2, :3] = image2
    show_img('level', image3)

    # 垂直合併
    image4 = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
    image4[:h1, :w1, :3] = image1
    image4[h1:h1 + h2, :w2, :3] = image2
    show_img('vertical', image4)
def 彩色圖像轉灰階(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (250, 250))
    show_img('gray1', image)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (250, 250))
    show_img('origin', image)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (250, 250))
    image1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    show_img('gray2', image1)
def 彩色圖像二值化(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (280, 280))
    show_img('gray1', image)

    ret1, mask1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    show_img('BINARY', mask1)
    ret2, mask2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    show_img('BINARY_INV', mask2)
    ret3, mask3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    show_img('TRUNC', mask3)
    ret4, mask4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    show_img('TOZERO', mask4)
    ret5, mask5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
    show_img('TOZERO_INV', mask5)

# 影像侵蝕：以Kernel(卷積核)
# 滑動進行卷積運算，使影像中白色區域縮小、黑色區域擴大。
# 用途：可用於去除影像中的噪點或加粗字跡。
def 影像侵蝕(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (300, 300))
    show_img('origin', image)

    kernel = np.ones((3, 3), np.uint8)
    erode_image = cv2.erode(image, kernel, iterations=1)
    show_img('erode_image', erode_image)
# 膨脹：以Kernel(卷積核)
# 滑動進行卷積運算，使影像中白色區域擴大、黑色區域縮小。
#
# 用途：可用於填補影像中的小孔洞或使字跡變細。
def 影像膨脹(image_path):
    # 顯示圖檔
    def show_img(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (300, 300))
    show_img('origin', image)

    kernel = np.ones((3, 3), np.uint8)
    dilate_image = cv2.dilate(image, kernel, iterations=1)
    show_img('dilate_image', dilate_image)
# 讀取圖片("image",'./大腦已超載.jpg')
# 存取圖片("./大腦已超載_新增圖片.jpg","大腦已超載.jpg",)
# 讀取影片("影片.mp4",3)
# 水平翻轉("level",'./大腦已超載.jpg')
# 垂直翻轉("vertical",'./大腦已超載.jpg')
# 影像旋轉_90度旋轉("./大腦已超載.jpg")
# 任意角度旋轉("./大腦已超載.jpg")
# 影像放大縮小("./大腦已超載.jpg")
# 比例縮放("./大腦已超載.jpg")
# 影像分割("./大腦已超載.jpg")
# 影像合併_圖像大小一致("./大腦已超載.jpg",'./peach.jpg')
# 影像合併_圖像大小不一致("./大腦已超載.jpg",'./peach.jpg')
# 彩色圖像轉灰階("./大腦已超載.jpg")
# 彩色圖像二值化("./大腦已超載.jpg")
# 影像侵蝕("./大腦已超載.jpg")
# 影像膨脹("./大腦已超載.jpg")