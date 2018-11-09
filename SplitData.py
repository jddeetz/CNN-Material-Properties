#This function splits the dataset into a training and test set

def SplitData(images,properties,split):
    
    #Import random number generator
    from random import random
    import numpy as np
    
    size=images[0].shape[0]
    
    X_train=images[0].reshape([size,size,size,1])
    X_test=images[1].reshape([size,size,size,1])
    y_train=np.array([properties.iloc[0,1]])
    y_test=np.array([properties.iloc[1,1]])
    
    for i in range(2,len(images)):
        if(random()>split):
            X_train=np.append(X_train,images[i].reshape([size,size,size,1]),axis=3)
            y_train=np.append(y_train,properties.iloc[i,1])
        else:
            X_test=np.append(X_test,images[i].reshape([size,size,size,1]),axis=3)
            y_test=np.append(y_test,properties.iloc[i,1])
    
    return X_train,X_test,y_train,y_test
            
    

