import numpy as np 
import cv2
import math


#get the image directory and block size from the user
image = input("Enter image directory : ") 
ex ='.'+ input("Enter image Extention (jpg,png,..) : ") 
image2=image+ex
SlidingWindowSize = int(input("Enter Sliding Window Size :"))
LookahedBufferSize = int(input("Enter Lookahed Buffer Size) :"))
Width = int(input("Enter the width of the image :"))
Height = int(input("Enter the height of the image :"))
SearchBufferSize = SlidingWindowSize-LookahedBufferSize

# Load an color image in grayscale
img = cv2.imread(image2,0)

#Flatten the image into vector
arr=np.array(img)
FlattenImg=arr.flatten()    

#Encoding
Enc =np.array([],dtype='uint8')
TempData = [None]*3
SlidingStart = 0
FirstElements= FlattenImg[0:SearchBufferSize]


while SlidingStart+SlidingWindowSize <= len(FlattenImg):
    LookaheadStart =  SearchBufferSize
    SearchStart =  SearchBufferSize - 1
    LookaheadPointer = LookaheadStart
    SearchPointer =SearchStart
    GoBack = 1
    length = 0
    SlidingWindow= FlattenImg[SlidingStart:SlidingStart+SlidingWindowSize]
    SearchWindow= FlattenImg[SlidingStart:SlidingStart+(SlidingWindowSize-LookahedBufferSize)]
    LookaheadgWindow= FlattenImg[SlidingStart+SlidingWindowSize-LookahedBufferSize:SlidingStart+SlidingWindowSize]
    while SearchPointer >=0 :
        while( LookaheadPointer < SlidingWindowSize-1 and SlidingWindow[SearchPointer]==SlidingWindow[LookaheadPointer] ):
            length+=1
            SearchPointer+=1
            LookaheadPointer+=1
        if  TempData[1] is None or length >= TempData[1]  :
            TempData[0] = GoBack if length !=0 else 0
            TempData[1] = length
            TempData[2] = SlidingWindow[LookaheadPointer] if LookaheadPointer < SlidingWindowSize else 0
        SearchPointer-=length 
        LookaheadPointer-=length
        GoBack+=1
        SearchPointer-=1
        length=0
    if TempData[1]==0:
        TempData[0] = 0
        TempData[1] = 0
        TempData[2] = SlidingWindow[LookaheadPointer]    
    Enc=np.append(Enc,TempData)    
    SlidingStart+=TempData[1]+1
    TempData[1]=0 
LastElements=FlattenImg[SlidingStart+SlidingWindowSize:]
np.save(image+'Enc',Enc)

#Decoding

Enc=np.load(image+'Enc.npy','r')

Dec = np.array([],dtype='uint8')
for item in FirstElements:
    Dec=np.append(Dec,item)
for i in range(0,len(Enc),3):
    if(Enc[i]==0):
        Dec=np.append(Dec,Enc[i+2])
        continue
    else:
        FirstMatchIndex=len(Dec)-int(Enc[i])
        length=int(Enc[i+1])
        for j in range(length):
            Dec=np.append(Dec,Dec[FirstMatchIndex])
            FirstMatchIndex+=1
        Dec=np.append(Dec,Enc[i+2])
for item in LastElements:
    Dec=np.append(Dec,item)
while len(Dec)!=Width*Height:
    if(len(Dec)>Width*Height):             
        Dec=np.delete(Dec,-1)  
    else:
        Dec=np.append(Dec,0)

Dec=np.reshape(Dec, (Width, Height),order='F')
Dec=Dec.T  

#Save the image .
cv2.imwrite(image+'Out'+ex, Dec)