#This function splits the dataset into a training and test set

def SplitData(images,properties,split):
    
    #Import random number generator
    from random import random
    
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    for i in range(len(images)):
        if(random()>split):
            X_train.append(images[i])
            y_train.append(properties.iloc[i,1])
        else:
            X_test.append(images[i])
            y_test.append(properties.iloc[i,1])
    
    return X_train,X_test,y_train,y_test
            
    

