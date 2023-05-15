&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="664" alt="Screenshot 2023-01-05 at 19 53 04" src="https://user-images.githubusercontent.com/58992981/210847477-5534faeb-0f11-426d-8d86-3ad194622982.png">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="765" alt="Screenshot 2023-01-05 at 19 58 21" src="https://user-images.githubusercontent.com/58992981/210848487-920c4440-d8ca-44c8-af0a-800346653c47.png">


-----------------------------------------------------------------------------------------------------------------------------------------------------------

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/9654d12b-9f95-435a-8b1b-089bf27fab4a)

* learn basic pytorch by inplement CNN on MNIST
1. first we copy and learn pyturch code from https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
2. we add document to the code and run the code and get success of 99% of the test group
3. we change the convolition from kernal 5 to 2 convolution of kernal 3 run the code and get success of 95% of the test group 
4. we serch on the web for cnn with convolution kernal of 3 we found the cnn in
 https://towardsdatascience.com/mnist-handwritten-digits-classification-using-a-convolutional-neural-network-cnn-af5fafbc35e9
and we implement it on the previos code and get success of 100% of the test group
![1_3DUs-90altOgaBcVJ9LTGg](https://user-images.githubusercontent.com/58992981/203141001-85860bfd-d0c5-4aaa-bca1-15c8d57c19a2.png)
-------------

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/22d35922-05f1-446c-a2f7-f5b79a9e3829)


* Implementing an Autoencoder in PyTorch
1. first we copy and learn pyturch code from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

The encoder and the decoder are neural networks that build the autoencoder model, as depicted in the following figure:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![0_b5eT77a_idC3v3BP](https://user-images.githubusercontent.com/58992981/213869707-c2d390b2-b0a8-4bb8-b9ce-da56740a69fb.png)

---
2. We made a comparison between the TRAIN LOSS vs TEST LOSS for EPOCH 1-100:

<img src="https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/9d52b080-e5a5-4e85-b3d0-601de1ee8588" alt="your image" width="800" height="400">

<img src="https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/cf638862-97a5-4cc7-b534-435d52050f12" alt="your image" width="800" height="400">

<img src="https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/a1c72d84-189d-4702-ae00-f1cd682996d5" alt="your image" width="800" height="400">

<img src="https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/f056b908-4419-4155-82b6-63370fb204f9" alt="your image" width="800" height="400">

<img src="https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/1f5add36-8305-42e1-bba2-8162d7063332" alt="your image" width="800" height="400">

<img src="https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/84e493c6-50bb-4819-abaf-63a6586be3f7" alt="your image" width="800" height="400">

<img src="https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/7e439932-6129-4a75-805f-7ab019811d3b" alt="your image" width="800" height="400">


---
3. We made a comparison between the minimum LOSS of each "LATENT SIZE" from the group {2,4,8,16,32} :
  
  <img width="778" alt="Screenshot 2023-01-05 at 20 08 22" src="https://user-images.githubusercontent.com/58992981/210850182-f3d52a1e-81b9-4988-a69a-af4b08beece2.png">
  
---

4. We compared the images created by the networks of each "Latent Size" :


![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/e1be3d2b-c9f6-47ae-9fd1-6a16b1ae3a2b)





---
5. We created a table of type cvs containing all the results of all sections 1 to 4, the link of Latent sizes statistic: https://docs.google.com/spreadsheets/d/1uFwPMJs6VD79z750eAqqNCx-qtuPh9zmGxM_9ECyZoQ/edit#gid=0

---

6. Adding a vector of random values to encoder output in autoencoder:
* For example Latent SIZE 64

<img width="708" alt="Screenshot 2023-01-05 at 20 41 13" src="https://user-images.githubusercontent.com/58992981/210856227-5f9b247d-6730-4eb9-b482-49d00d0e5ae7.png">

7. Testing the effect of changing coordinates in the Latent vector on the image created by the Decoder:
* In each image we changed each of its coordinates in the range of (1,1-) in increments of 0.25
So that you can see how changing each of the coordinates affects the image,
For example Latent SIZE is 16 and Number is 2

<img width="947" alt="Screenshot 2023-01-05 at 20 49 55" src="https://user-images.githubusercontent.com/58992981/210859926-48a3e8e7-2ffc-454c-bef3-8df73001f6f2.png">

8. We have converted an image to another image, Let's take an example when Latent SIZE is 64 and conversion from 2 to 7 :

<img width="815" alt="Screenshot 2023-01-05 at 20 58 36" src="https://user-images.githubusercontent.com/58992981/210860172-45af5d79-d23c-4f70-9fa3-85356ed55de2.png">

---

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/0e790ab3-09e7-4368-93ad-e67a676e87e4)

Variational Autoencoder (VAE) is a type of neural network that learns to encode input data into a lower-dimensional representation, called a latent variable, and decode the latent variable back into the original input data. VAEs use a probabilistic approach to training and can generate new data samples similar to the original input data. They have a wide range of applications, including data compression, anomaly detection, and data generation, Beta is a hyperparameter in Variational Autoencoders (VAE) that controls the level of importance given to the regularization term in the loss function. The regularization term encourages the latent variables to follow a prior distribution, typically a Gaussian distribution, and is measured by the Kullback-Leibler (KL) divergence. A higher value of beta places more emphasis on the regularization term, resulting in a more compressed latent space, whereas a lower value of beta places more emphasis on the reconstruction loss and can lead to overfitting.

---
We implemented the code and performed several experiments:
![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/96c236e0-7b56-4017-8466-b710a16b5f22)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/f463c9fc-a45b-486f-b4e7-f7c81a69bc38)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/31872fa7-7b9d-40b9-bc6a-d8097148dd9f)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/56f8f3b2-14c3-4967-b596-53ec413457e3)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/a362561c-b4e8-41b1-a3aa-a740b0540ac9)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/49fbf9d4-bb8e-48cd-90e2-7947f2403cdf)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/162eb95a-edbc-48b6-85d1-8f00db0a5da8)

All Variational Autoencoder experiments, which involve encoding and decoding input data into a lower-dimensional latent variable space, have been saved to a file for future reference. In addition, t-SNE analysis, which visualizes the high-dimensional data in a two -dimensional space, has also been saved to the file to facilitate easy comparison and analysis of the results.
The Link : https://docs.google.com/presentation/d/1QPu-V9ZM8QYiV423T-WDm_yTfunkIeFG/edit#slide=id.p8


