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

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/9d52b080-e5a5-4e85-b3d0-601de1ee8588)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/cf638862-97a5-4cc7-b534-435d52050f12)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/a1c72d84-189d-4702-ae00-f1cd682996d5)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/f056b908-4419-4155-82b6-63370fb204f9)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/1f5add36-8305-42e1-bba2-8162d7063332)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/84e493c6-50bb-4819-abaf-63a6586be3f7)

![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/7e439932-6129-4a75-805f-7ab019811d3b)


---
3. We made a comparison between the minimum LOSS of each "LATENT SIZE" from the group {2,4,8,16,32} :
  
  <img width="778" alt="Screenshot 2023-01-05 at 20 08 22" src="https://user-images.githubusercontent.com/58992981/210850182-f3d52a1e-81b9-4988-a69a-af4b08beece2.png">
  
---

4. We compared the images created by the networks of each "Latent Size" :


![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/e6e3f659-a3e0-4c12-bcaa-f0aea60e2600)
![image](https://github.com/benaa1995/Vector-Quantized-Variational-Autoencoder/assets/58992981/5860edc9-39c4-4a20-91d7-02c2af91ffc7)
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

* implemant vq-vae 2 
* Stage 1 - reconstruct image
1. This algorithm get an image X passing X through "bottom encoder" and get the output Z_bottom
2. Then passing Z_b through "top encoder" and get the output Z_top
3. Replacing Z_top with the matching code book vectors from the "top code book" and get the output E_top
4. passing E_top through "top decoder" and get the output dec_top
5. Adding Z_b + dec_top and replacing them with the matching code book vectors from the "bottom code book" 
and get the output E_bottom
6. Adding E_bottom + E_top and passing then through the bottom encoder and getting X_rec

* Stage 2 - generate image 
1. Convert all the dataset from image X to Z_bottom, Z_top and label (if we have label) using the
trained "bottom encoder" and "top encoder" from the reconstruction task.
2. Choose a "pixel cnn" model.
3. Train pixel_cnn_top with (Z_top, label) from the new dataset
4. Train pixel_cnn_bottom with (Z_bottom, label) from the new dataset
5. Generate E_top using our trained pixel_cnn_top
6. Generate E_bottom conditionally on E_top using our trained pixel_cnn_bottom 
7. Adding E_top + E_bottom and passing then through the trained "bottom encoder" and getting X_rec as our 
generate image


* train the model
- Our loss function combine of three parts:
1. Reconstruct loss - Simple L2 between the original image X end reconstruct imag X_REC

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This loss will propagate the bottom decoder
then it will skip the code book directly to the encoder because
code book is not continuous
2. Encoder loss - subtract Z from E
3. Code book loss -  subtract E from Z
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
There is more advanced loss for the code book. Its call "moving average", this loss save the average loss for each vector e in the "code book"
and the average times vector e appear in E. Then calculate the current loss and appearance of each vector e.
Then adding the total appearance and total loss (with sum decay) and calculate the loss by dived the total loss on the total appearance.  



