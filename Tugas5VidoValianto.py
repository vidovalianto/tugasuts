import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

bias= 0.2
alpha= 0.1

k=5
epoch = 100

mdteta = np.zeros(shape=(4,1))
mtetab = np.zeros(shape=(4,1))
totalerrortrainkfold= np.zeros(shape=(epoch,1))
totalerrortestkfold= np.zeros(shape=(epoch,1))

num_of_output = 2
layers = 2
neurons = [2,num_of_output]

error_i = np.zeros(num_of_output)

h = np.zeros((2,2))
out = np.zeros((2,2))

dataframe = pd.read_csv("./data.csv")
dataframe = dataframe['teta'].str.split(',', expand=True)

def nilaih(x,teta,bias):
    return (np.dot(dataframe.iloc[x, :-1],np.transpose(teta))+bias)

def sigm(hasilh):
    try:
        ans = (1/(1+math.exp( -hasilh )))
    except OverflowError:
        ans = float('inf')
    return ans

   
def class_label1(row):
    if row[4] == 'Iris-virginica':
        return 1
    else:
        return 0

def class_label2(row):
    if row[4] == 'Iris-versicolor':
        return 1
    else:
        return 0
        
def class_label3(row):
    if row[4] == 'Iris-virginica':
        return 1
    else:
        return 0
    
def mse(err):
    return ((np.dot(err, np.transpose(err)))/len(err))

def delta(g, y, x):
    return (2*(g-y)*(1-g)*g*x)


dataframe[5] = dataframe.apply(lambda row: class_label1(row), axis=1)
dataframe[6] = dataframe.apply(lambda row: class_label2(row), axis=1)
dataframe[7] = dataframe.apply(lambda row: class_label3(row), axis=1)

dataframe[5] = dataframe[5].astype('float64')
dataframe[6] = dataframe[6].astype('float64')
dataframe[7] = dataframe[6].astype('float64')



for fold in range(k):
    start = 0
    end = 0
    if (fold == 0):
        testdataframe = np.vstack([dataframe.iloc[0:50, 0:4]])
        testlabel = np.vstack([dataframe.iloc[0:50, 5:7]])
        traindataframe = np.vstack([dataframe.iloc[50:150, 0:4]])
        trainlabel = np.vstack([dataframe.iloc[50:140, 5:7]])
    elif (fold == 4):
        testdataframe = np.vstack([dataframe.iloc[120:150, 0:4]])
        testlabel = np.vstack([dataframe.iloc[120:150, 5:7]])
        traindataframe = np.vstack([dataframe.iloc[0:120, 0:4]])
        trainlabel = np.vstack([dataframe.iloc[0:120, 5:7]])
    else:
        start = ((len(dataframe)//k)*fold)
        end = start + (len(dataframe)//k)
        testdataframe = np.vstack([dataframe.iloc[start:end, 0:4]])
        testlabel = np.vstack([dataframe.iloc[start:end, 5:7]])
        traindataframe = np.vstack([dataframe.iloc[0:start, 0:4], dataframe.iloc[end:150,0:4]])
        trainlabel = np.vstack([dataframe.iloc[0:start, 5:7], dataframe.iloc[end:150,5:7]])
        
    train_count = len(trainlabel)
    test_count = len(testlabel)
    
    new_teta = np.array([[[0.1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]])
    new_bias = np.array([[0.1,0.1], [0.1,0.1]])

    for n in range(epoch):

        localerror = 0.0000
        
        for i in range(train_count):

            tetas = new_teta.copy()
            biases = new_bias.copy()


            for layer in range(layers):
                type(layer)

                for neuron in range(neurons[layer]):

                    teta = tetas[layer,neuron]
                    bias = biases[layer,neuron]

                    if (layer == 0):
                        h[layer, neuron] = nilaih(traindataframe[i], teta, bias)
                    else:
                        h[layer, neuron] = nilaih(out[layer-1], teta, bias)

                    out[layer, neuron] = sigm(h[layer, neuron])


         
            for idx in range(num_of_output):
                error_i[idx] = (out[1,idx] - trainlabel[i, idx])

            dteta = np.array([[[1, 1, 1, 1],[1, 1, 1, 1]], [[1,1], [1,1]]])
            dbias = np.zeros((2,2))

            for layer in range(layers-1, -1, -1):

                for out_neuron in range(neurons[layer]-1, -1, -1):

                   
                    if (layer != 0):   
                        dbias[layer][out_neuron] = (error_i[out_neuron]*out[layer,out_neuron]*(1-out[layer,out_neuron]))

                        for in_neuron in range(neurons[layer-1]-1, -1, -1):                
                            dteta[layer][out_neuron][in_neuron] = (dbias[layer][out_neuron]*out[layer-1][in_neuron])

                            new_teta[layer][out_neuron][in_neuron] = (tetas[layer][out_neuron][in_neuron] -(alpha*dteta[layer][out_neuron][in_neuron]))


                    else:
                        dbias[layer][out_neuron] = ((dbias[layer+1][0]*tetas[layer+1][0][out_neuron]+dbias[layer+1][1]*tetas[layer+1][1][out_neuron])*out[layer][out_neuron]*(1-out[layer][out_neuron]))
                        for in_neuron in range(len(traindataframe[i, :])-1, -1, -1):
                            dteta[layer][out_neuron][in_neuron] = (dbias[layer][out_neuron]* traindataframe[i,in_neuron])
                            new_teta[layer][out_neuron][in_neuron] = (tetas[layer][out_neuron][in_neuron] - (alpha*dteta[layer][out_neuron][in_neuron]))

                  
                    new_bias[layer][out_neuron] = biases[layer][out_neuron] - (alpha*dbias[layer][out_neuron])

            localerror += mse(error_i)


        totalerrortrainkfold[n] += (localerror/(len(traindataframe)))

        localerror = 0.0000

        
        for j in range(test_count):

            tetas = new_teta.copy()
            biases = new_bias.copy()

            for layer in range(layers):

                for neuron in range(neurons[layer]):

                    teta = tetas[layer,neuron]
                    bias = biases[layer,neuron]

                    if (layer == 0):
                        h[layer, neuron] = nilaih(testdataframe[j], teta, bias)
                    else:
                        h[layer, neuron] = nilaih(out[layer-1], teta, bias)

                    out[layer, neuron] = nilaih(h[layer, neuron])

            for idx in range(num_of_output):
                error_i[idx] = (out[1,idx] - trainlabel[j, idx])

            localerror += mse(error_i)

        totalerrortestkfold[n] += (localerror/test_count)

    print(totalerrortestkfold)
                 

x = np.arange(epoch)
y1 = totalerrortrainkfold.copy()
y2 = totalerrortestkfold.copy()

plt.figure(figsize=(12,8))
plt.plot(x,y1, color="green")
plt.plot(x,y2, color="red")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Grafik Error setiap Epoch")
plt.legend(["dataframe Training", "dataframe Validation"])
plt.grid()
plt.show()