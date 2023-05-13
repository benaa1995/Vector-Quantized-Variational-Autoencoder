&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="664" alt="Screenshot 2023-01-05 at 19 53 04" src="https://user-images.githubusercontent.com/58992981/210847477-5534faeb-0f11-426d-8d86-3ad194622982.png">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="765" alt="Screenshot 2023-01-05 at 19 58 21" src="https://user-images.githubusercontent.com/58992981/210848487-920c4440-d8ca-44c8-af0a-800346653c47.png">


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


2. We made a comparison between the TRAIN LOSS vs TEST LOSS for EPOCH 1-100:

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/9d52b080-e5a5-4e85-b3d0-601de1ee8588)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/cf638862-97a5-4cc7-b534-435d52050f12)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/a1c72d84-189d-4702-ae00-f1cd682996d5)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/f056b908-4419-4155-82b6-63370fb204f9)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/1f5add36-8305-42e1-bba2-8162d7063332)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/84e493c6-50bb-4819-abaf-63a6586be3f7)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/7e439932-6129-4a75-805f-7ab019811d3b)



3. We made a comparison between the minimum LOSS of each "LATENT SIZE" from the group {2,4,8,16,32} :
  
  <img width="778" alt="Screenshot 2023-01-05 at 20 08 22" src="https://user-images.githubusercontent.com/58992981/210850182-f3d52a1e-81b9-4988-a69a-af4b08beece2.png">

4. We compared the images created by the networks of each "Latent Size" :
* Let's take number 2 for example

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="298" alt="Screenshot 2023-01-05 at 20 26 30" src="https://user-images.githubusercontent.com/58992981/210853583-1e193681-79d5-4183-9249-361ab6fb9164.png">

<img width="674" alt="Screenshot 2023-01-05 at 20 23 19" src="https://user-images.githubusercontent.com/58992981/210853317-6be1b1bd-5cbd-42d0-a71b-130f425e8871.png">

5. We created a table of type cvs containing all the results of all sections 1 to 4, the link of Latent sizes statistic: https://docs.google.com/spreadsheets/d/1uFwPMJs6VD79z750eAqqNCx-qtuPh9zmGxM_9ECyZoQ/edit#gid=0

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

* learn basic pytorch by inplement CNN on MNIST
1. first we copy and learn pyturch code from https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
2. we add document to the code and run the code and get success of 99% of the test group
3. we change the convolition from kernal 5 to 2 convolution of kernal 3 run the code and get success of 95% of the test group 
4. we serch on the web for cnn with convolution kernal of 3 we found the cnn in
