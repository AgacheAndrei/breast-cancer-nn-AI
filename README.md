# Neuronal Network for breast cancer prediction - AI
## Chosen topic 
Breast tumor classification into malignant or benign tumors. The data set is in a .csv format

## Solution 
### Programing languages and technology used

<img align="left" width="30px" style="padding-right:10px" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" /> 
<img align="left" alt="keras" width="30px" style="padding-right:10px" src="https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/20ce22c5-31f8-473f-9cf0-15e7743c3dc0"/> 
<img align="left" alt="scikit learn" width="60px" style="padding-right:10px" src="https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/e1ef909a-bc2d-4662-9c22-09df48be9f05"/> 

<br>

### Model
<pre>
model = Sequential()
model.add(Dense(70, input_dim = 30, activation = "relu"))
model.add(Dense(1, activation = "linear"))
model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
</pre>
### Training accuracy
<pre>
test_loss, test_accuracy = model.evaluate(xtest, ytest) 
print("Accuracy: %.2f " %(test_accuracy * 100))
6/6 [==============================] - 0s 2ms/step - loss: 0.0451 - accuracy: 0.9532 
Accuracy: 96.33
</pre>
### Test accuracy
![nn_test_accuracy](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/9f75f00a-fdec-436a-9023-2e7dba1ec0f6)
### Photos from the run of the project 
#### File upload
![NN_1](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/e71c2d41-9ae5-4fb6-95bd-87652a365650)
#### Drop the unnecessary data
![NN_2](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/57cae67c-eed1-4269-8c51-499c854c34b7)
![NN_3](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/5b68c87a-a4f4-49f1-a808-ad42bf3c6653)
#### Change alfanumeric data to numeric data
![NN_4](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/bf6d0de0-0721-4d64-ab33-8c50c5be1b89)
#### Change the data type from frame to float64 and droping the diagnosis column
![NN_5](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/37862936-e43f-4ef0-8e14-9946b25243f5)
#### The data is stacked in a vertical array
![NN_6](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/e390ba25-7987-4aa0-8b83-f0786335c907)
#### Transpose the vertical array
![NN_7](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/a29341a8-63d0-449c-87c0-4d659cb60dc3)
#### The model and the neural network structure
![NN_8](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/08979f73-1d3b-4660-9655-02fa19368cf6)
#### The model compiles with the chosen parameters and after that fits the training data in 100 epochs
![NN_9](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/89cae96f-8906-4f95-9cb1-c3de5fc29ead)
#### Test the accuracy, the model predicts the output, the mean squared error is calculated, with sklearn.metrics we see the classification raport and confusion matrix using test and predicted data
![NN_10](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/3d1318c1-f809-48bf-9f77-41ebd4db3548)
#### Error graph of the AI model
![NN_11](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/2f796264-447a-470d-9967-bbda531bc64f)
#### Confusion matrix and accuracy using test and predicted data
![NN_12](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/f46a3d92-1b11-46c7-8db0-589254400726)
#### Barchart on the OX -> 0.0 represent Bening / 1.0 represent Maligne || OY -> Number of people
![NN_13](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/cbd2d27f-8d85-4053-b078-b3b7eea6f940)
#### Charts with the data, showing difference in the parameters of people with a Maligne tumor vs people with a Bening tumor
![NN_14](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/cdb011aa-559b-4460-82bd-1311fc89efda)
![NN_15](https://github.com/AgacheAndrei/breast-cancer-nn-AI/assets/36128809/b6d0eadd-1e77-46f7-a80f-23b60ca19fb1)
