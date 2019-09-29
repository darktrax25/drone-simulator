import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os

def keys_to_output(keys):
    #[Q,W,E]
    #[]
    output = [0,0,0]
    
    if 'Q' in keys:
        output[0] = 1
    elif 'E' in keys:
        output[2] = 1
    else:
        output[1] = 1
        
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    last_time = time.time()
    
    while True:
        screen =  grab_screen(region=(0,40,800,640))
        cv2.imshow('window2',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(80,60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        #print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        
        
        if cv2.waitKey(25) & 0xFF == ord('p'):
            cv2.destroyAllWindows()
            print(len(training_data))
            np.save(file_name,training_data)
            break

#        if len(training_data) % 500 == 0:
#            print(len(training_data))
#            np.save(file_name,training_data)
            
            
        
        
main()