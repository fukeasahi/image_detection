#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:





# In[2]:


import cv2
from IPython.display import display, Image

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cascade_path = "./opencv-master/data/haarcascades/haarcascade_frontalcatface.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    
    color = (255,255,255)
    
    while True:
        ret, frame = cap.read()
        facerect = cascade.detectMultiScale(frame,scaleFactor=1.2, minNeighbors=2,minSize=(10,10))
        
        if  len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),color)
                
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()


# In[20]:



# # VideoCapture オブジェクトを取得します
# capture = cv2.VideoCapture(0)

# while(True):
#     ret, frame = capture.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




