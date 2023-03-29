import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression


# import seaborn as sns

img_idx_MNIST_adapter = lambda tar, n: {i: np.where(tar == i)[0][0] for i in range(n)}
def img_idx_CIFAR10_adapter(targets, n):
    t_idx = {}
    for i in range(n):
        for j in range(len(targets)):
            if i == targets[j]:
                t_idx[i] = j
                break
    return t_idx


plot_MNIST_adapter = lambda img: img

plot_CIFAR10_adapter = lambda img: np.transpose(img, (1, 2, 0))

targets_MNIST_adapter = lambda test_dataset: test_dataset.targets.numpy()

targets_CIFAR10_adapter = lambda test_dataset: test_dataset.targets
def plot_ae_outputs(encoder, decoder, test_dataset, device, targets_adapter, img_idx_adapter, plot_adapter,
                    cmap=None, n=10):
    plt.figure(figsize=(16, 4.5))
    # get index of images from every target's type
    targets = targets_adapter(test_dataset)
    t_idx = img_idx_adapter(targets, n)
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        # get the original image
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        # get the re-construct image
        with torch.no_grad():
            encoded_data, _, _ = encoder(img)
            rec_img = decoder(encoded_data)
        # plot the original image
        img = img.cpu()
        np_img = img.cpu().squeeze().numpy()
        plt.imshow(plot_adapter(np_img), cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        # plot the re-construct image
        rec_img = rec_img.cpu()
        np_rec_img = rec_img.cpu().squeeze().numpy()
        plt.imshow(plot_adapter(np_rec_img), cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.show()




# save the latent vector in cvs and train liniar logistic on this model
def convert_latent_to_cvs(encoder, latent_size, file_name, dataloader, device):
    req_col = {}
    X = "X"
    for i in range(latent_size):
        temp_col = X + str(i + 1)
        req_col[temp_col] = []
    req_col["Y"] = []

    # convert to data frae
    df = pd.DataFrame.from_dict(req_col)
    encoder.eval()
    with torch.no_grad():  # No need to track the gradients
        for image_batch, target_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            curr_batch_size = image_batch.shape
            curr_num_of_row = curr_batch_size[0]
            # Encode data
            encoded_data, _, _ = encoder(image_batch)
            encoded_data = encoded_data.to(device).detach().numpy()
            target_batch = target_batch.to(device).detach().numpy()
            rows = np.zeros((curr_num_of_row, latent_size + 1))
            rows[:, :-1] = encoded_data
            bach_y = target_batch.reshape((-1, 1))
            rows[:, -1:] = bach_y
            # add the row to the data frame
            for row in rows:
                df.loc[len(df)] = row
        print(df)
        file_name = file_name + '.cvs'
        # save the dataframe as cvs file
        df.to_csv(file_name)


def train_with_log_reg(test_path, train_path):
    # load the data
    # df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
    x_test = df_test.to_numpy()
    x_test = x_test[:, 1:-1]
    y_test = df_test['Y'].values
    x_train = df_train.to_numpy()
    x_train = x_train[:, 1:-1]
    y_train = df_train['Y'].values

    # sklrn linear regration
    # two calasses
    clf_1 = LogisticRegression(max_iter=1000)
    clf_1.fit(x_train, y_train)
    pred_y_train = clf_1.predict(x_train)
    pred_y_test = clf_1.predict(x_test)
    accuracy_test = 0
    accuracy_train = 0
    for i in range(len(y_train)):
        if pred_y_train[i] == y_train[i]:
            accuracy_train += 1
    for i in range(len(y_test)):
        if pred_y_test[i] == y_test[i]:
            accuracy_test += 1
    print("accuracy on train = ", 100 * accuracy_train / len(y_train), "%")
    print("accuracy on test = ", 100 * accuracy_test / len(y_test), "%")
