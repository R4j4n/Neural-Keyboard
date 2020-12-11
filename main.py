
#imports
import cv2
import dlib
import numpy as np
from math import hypot
from imutils import face_utils
from utils import *
import winsound


# KEYBOARD SETTING/VARIABLES: 
keyboard=np.zeros((300,800,3),np.uint8)
first_col_index=[0,10,20,30,40,50]
second_col_index=[1,11,21,31,41,51]
third_col_index=[2,12,22,32,42,52]
fourth_col_index=[3,13,23,33,43,53]
fifth_col_index=[4,14,24,34,44,54]
sixth_col_index=[5,15,25,35,45,55]
seventh_col_index=[6,16,26,36,46,56]
eighth_col_index=[7,17,27,37,47,57]
ninth_col_index=[8,18,28,38,48,58]
tenth_col_index=[9,19,29,39,49,59]
key_set={0:"1",1:"2",2:"3",3:"4",4:"5",5:"6",6:"7",7:"8",8:"9",9:"0",
            10:"q",11:"w",12:"e",13:"r",14:"t",15:"y",16:"u",17:"i",18:"o",19:"p",
            20:"a",21:"s",22:"d",23:"f",24:"g",25:"h",26:"j",27:"k",28:"l",29:";",
            30:"z",31:"x",32:"c",33:"v",34:"b",35:"n",36:"m",37:"<",38:">",39:"?",
            40:"+",41:"-",42:",",43:".",44:"/",45:"*",46:"@",47:".",48:"!",49:" ",
            50:"->",51:"->",52:"->",53:"->",54:"->",55:"->",56:"->",57:"->",58:"->",59:"->",}
#counters
frame_count_column=0 #this is frame count for column
frame_count_row=0 #this is frame count for row
col_index=[]
col=0
blink_count=0 #this for couting the blink (used for blink for changing the row and column)
blink_count_indivisual_key=0 #this is for counting the blink to check whether the key should press or not
font_letter=cv2.FONT_HERSHEY_PLAIN
col_select=False #this for selecting the particular column
row=0 #this is to count the row after particular column is selected
IMG_SIZE=(34,26)
###################
type_text="" #this is to store the typed character 
###################

#user defined class object for blink detection using cnn model
bd=Blink_detection()
white_board=np.ones((100,800,3),np.uint8)

#######################
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("FILES\\shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
#####################

#FUNCTION TO DRAW KEYBOARD
def draw_keyboard(letter_index,letter,light):
    if letter_index==0:
        x=0
        y=0
    elif letter_index==1:
        x=80
        y=0
    elif letter_index==2:
        x=160
        y=0
    elif letter_index==3:
        x=240
        y=0
    elif letter_index==4:
        x=320
        y=0
    elif letter_index==5:
        x=400
        y=0
    elif letter_index==6:
        x=480
        y=0
    elif letter_index==7:
        x=560
        y=0
    elif letter_index==8:
        x=640
        y=0
    elif letter_index==9:
        x=720
        y=0
    elif letter_index==10:
        x=0
        y=50
    elif letter_index==11:
        x=80
        y=50
    elif letter_index==12:
        x=160
        y=50
    elif letter_index==13:
        x=240
        y=50
    elif letter_index==14:
        x=320
        y=50
    elif letter_index==15:
        x=400
        y=50
    elif letter_index==16:
        x=480
        y=50
    elif letter_index==17:
        x=560
        y=50
    elif letter_index==18:
        x=640
        y=50
    elif letter_index==19:
        x=720
        y=50
    elif letter_index==20:
        x=0
        y=100
    elif letter_index==21:
        x=80
        y=100
    elif letter_index==22:
        x=160
        y=100
    elif letter_index==23:
        x=240
        y=100
    elif letter_index==24:
        x=320
        y=100
    elif letter_index==25:
        x=400
        y=100
    elif letter_index==26:
        x=480
        y=100
    elif letter_index==27:
        x=560
        y=100
    elif letter_index==28:
        x=640
        y=100
    elif letter_index==29:
        x=720
        y=100
    elif letter_index==30:
        x=0
        y=150
    elif letter_index==31:
        x=80
        y=150
    elif letter_index==32:
        x=160
        y=150
    elif letter_index==33:
        x=240
        y=150
    elif letter_index==34:
        x=320
        y=150
    elif letter_index==35:
        x=400
        y=150
    elif letter_index==36:
        x=480
        y=150
    elif letter_index==37:
        x=560
        y=150
    elif letter_index==38:
        x=640
        y=150
    elif letter_index==39:
        x=720
        y=150
    elif letter_index==40:
        x=0
        y=200
    elif letter_index==41:
        x=80
        y=200
    elif letter_index==42:
        x=160
        y=200
    elif letter_index==43:
        x=240
        y=200
    elif letter_index==44:
        x=320
        y=200
    elif letter_index==45:
        x=400
        y=200
    elif letter_index==46:
        x=480
        y=200
    elif letter_index==47:
        x=560
        y=200
    elif letter_index==48:
        x=640
        y=200
    elif letter_index==49:
        x=720
        y=200
    elif letter_index==50:
        x=0
        y=250
    elif letter_index==51:
        x=80
        y=250
    elif letter_index==52:
        x=160
        y=250
    elif letter_index==53:
        x=240
        y=250
    elif letter_index==54:
        x=320
        y=250
    elif letter_index==55:
        x=400
        y=250
    elif letter_index==56:
        x=480
        y=250
    elif letter_index==57:
        x=560
        y=250
    elif letter_index==58:
        x=640
        y=250
    elif letter_index==59:
        x=720
        y=250
    font=cv2.FONT_HERSHEY_PLAIN
    letter_thickness=2
    key_space=2
    font_scale=3
    height=50
    width=80
    if light==True:
        cv2.rectangle(keyboard,(x+key_space,y+key_space),(x+width-key_space,y+height-key_space),(0,255,0),-1)
    else:
        cv2.rectangle(keyboard,(x+key_space,y+key_space),(x+width-key_space,y+height-key_space),(0,255,0),key_space)
    letter_size=cv2.getTextSize(letter,font,font_scale,letter_thickness)[0]
    letter_height,letter_width=letter_size[1],letter_size[0]
    letter_x=int((width-letter_width)/2)+x
    letter_y=int((height+letter_height)/2)+y
    cv2.putText(keyboard,letter,(letter_x,letter_y),font,font_scale,(255,255,255),letter_thickness)

while True:
    main_windows = np.zeros((780,1000,3),np.uint8)
    if col_select==True:
        frame_count_row=frame_count_row+1
    else:
        frame_count_column=frame_count_column+1
    
    if frame_count_column==10:
        col=col+1
        if col==10:
            col=0 #reseting the column
        frame_count_column=0
    if frame_count_row==10:
        row=row+1
        if row==6:
            row=0 #resetting the row
            col_select=False
        frame_count_row=0
    if col==0:
        col_index=first_col_index
    elif col==1:
        col_index=second_col_index
    elif col==2:
        col_index=third_col_index
    elif col==3:
        col_index=fourth_col_index
    elif col==4:
        col_index=fifth_col_index
    elif col==5:
        col_index=sixth_col_index
    elif col==6:
        col_index=seventh_col_index
    elif col==7:
        col_index=eighth_col_index
    elif col==8:
        col_index=ninth_col_index
    elif col==9:
        col_index=tenth_col_index
    keyboard[:]=(0,0,0) #reseting the keyboard
    if col_select==False:
        for i in range(0,60):
            if i in col_index:
                draw_keyboard(i,key_set[i],True)
            else:
                draw_keyboard(i,key_set[i],False)
    else:
        for i in range(0,60):
            if i == col_index[row]:
                draw_keyboard(i,key_set[i],True)
                
            else:
                draw_keyboard(i,key_set[i],False)

    #Blink integration begin
    _,frame=cap.read() #reading the frame form the webcam
    
    
    frame = cv2.flip(frame, flipCode=1 )
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #coverting the bgr frame to gray scale
    faces=detector(gray) #this returns the dlib rectangle
    #now extracting the rectangle which contain the upper and lower cordinates of the face
    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = bd.crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = bd.crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

       


        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l,pred_r=bd.model_predict(eye_input_l,eye_input_r)

        if pred_l < 0.1 and pred_r <0.1:
            cv2.putText(main_windows,"------BLINK DETECTED------",(375,325), font_letter,1, (0,0,255),2)

            print("blink detected")
            blink_count=blink_count+1
            if col_select==True:
                blink_count_indivisual_key=blink_count_indivisual_key+1
                frame_count_row=frame_count_row-1
            else:
                frame_count_column=frame_count_column-1
        else:
            blink_count=0
            blink_count_indivisual_key=0
        if blink_count==10:
            col_select=True
        #implementing keyboard typing
        if blink_count_indivisual_key==10 and col_select==True:
            col_select=False #to disable the active column
            type_text=type_text+key_set[col_index[row]]
            blink_count_indivisual_key=0
            white_board[:]=(0,0,0)
            winsound.Beep(500,100)
            cv2.putText(white_board,type_text,(10,50),cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),3)
            row=0 #resetting the row


        # visualize
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(64,224,208), thickness=2)
        cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,0,0), thickness=2)

        # Combaining all windows into single window: 
        main_windows[50:150, 100:200] = cv2.resize(cv2.cvtColor(eye_img_l,cv2.COLOR_BGR2RGB),(100,100))
        cv2.putText(main_windows,"LEFT EYE",(100,170), font_letter,1, (0,255,0),1)
        cv2.putText(main_windows,str(state_l+"%"),(100,200), font_letter,2, (0,0,255),1)
        main_windows[50:150, 800:900] = cv2.resize(cv2.cvtColor(eye_img_r,cv2.COLOR_BGR2RGB),(100,100))
        cv2.putText(main_windows,"RIGHT EYE",(800,170), font_letter,1, (0,255,0),1)
        cv2.putText(main_windows,str(state_r+"%"),(800,200), font_letter,2, (0,0,255),1)

        main_windows[0:300, 300:700]= cv2.resize(frame,(400,300))
        main_windows[350:650, 100:900] =  keyboard
        main_windows[670:770, 100:900] = white_board
        cv2.imshow("Main_Windows",main_windows)
    key=cv2.waitKey(10)
    if key==ord('q'):
        break
cv2.destroyAllWindows()
