import cv2,numpy,os
alg="haarcascade_frontalface_default.xml"# importing algorithm
face_cascade=cv2.CascadeClassifier(alg)# reading & storing the algorithm in a variable
datasets="datasets"
(images,labels,names,id)=([],[],{},0)# intialising the variables labels=names,number of photos =images, id = name )
for (subdirs, dirs, files) in os.walk(datasets):# accesing the folder datasets 
    for subdir in dirs:# accesing sub-folders within main folder
        names[id] = subdir# getting names of the subfolders which is getting accesed 
        subjectpath = os.path.join(datasets, subdir)# creating path to acces folder containing images
        for filename in os.listdir(subjectpath):# to read the images in the sub folders
            path=subjectpath+'/'+filename# segregating the paths
            label=id
            images.append(cv2.imread(path,0))# read each image in the subfolder and append it into a a empty list images
            labels.append(int(label))#append all the names accesed so far
            # print (labels)
        id+=1
(width,height)=(130,100)# resizing the image
(images,labels)=[numpy.array(lis) for lis in [images,labels]]#converts images, labels into matrix format
#print(images,labels)# prints convrted images into array(pixels)
# load recognizers
model=cv2.face.LBPHFaceRecognizer_create()
#model=cv2.face.FisherFaceRecognizer_create()
# training or ML
model.train(images,labels)
print("training completed")
#face dtection algorithm
webcam=cv2.VideoCapture(0)
while True:
    (_,im)=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
            face=gray[y:y+h,x:x+w]
            face_resize=cv2.resize(face,(width, height))
            prediction=model.predict(face_resize)
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
            if prediction[1]<800:# 1st return value represents the id /class
                cv2.putText(im,'%s-%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
                print(names[prediction[0]])
                cnt=0
            else:
                cnt+=1
                cv2.putText(im,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
                if(cnt>100):
                      print("Unknown Person")
                      cv2.imwrite("imput.jpg")
                      cnt=0
    cv2.imshow('opencv',im)
    key=cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()
    
