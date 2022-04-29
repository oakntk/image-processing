import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

####สร้าง dataset
def create_dataset(img,id,img_id): 
    cv2.imwrite("data/id_"+str(id)+"_"+"pic"+"_"+str(img_id)+".jpg",img)

####โชวกล่อง
def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h), color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        coords = [x,y,w,h]
    return img,coords

####detect face
def detect(img,faceCascade,img_id):
    img,coords = draw_boundary(img,faceCascade,1.1,10,(0,0,255),"Face")
    ####check that face detection is active by check length of coordinate x,y,w and h
    if len(coords) == 4 :
        ####img(y:y+h, x:x+w)
        result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]

        #id of traning person
        id = 2
        
        create_dataset(result,id,img_id)
    return img


img_id = 0 #### global variable - set initial
cap = cv2.VideoCapture("kaew.mp4")


while (True):
    ret,frame = cap.read()
    frame = detect (frame,faceCascade,img_id) #### ส่งค่า
    cv2.imshow('Video',frame)
    img_id+=1
    if(cv2.waitKey(1) & 0xFF== ord('x')):
        break

cap.release()
cv2.destroyAllWindows