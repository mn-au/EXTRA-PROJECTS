import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class svm_():
    def __init__(self, learning_rate, epoch, C_value, X=None, Y=None, threshold=1e-3):
        # initializing SVM parameters and weights
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate  # step size for weight updates
        self.epoch = epoch  # number of iterations to train
        self.C = C_value  # regularization strength
        self.threshold = threshold  # early stopping threshold for loss change
        self.weights = None
        self.weights_at_early_stop = None
        # if input data is given, initialize weights with small random values
        if X is not None:
            self.weights = np.random.randn(X.shape[1]) * 0.01

    def pre_process(self):
        # standardizing input features (zero mean, unit variance)
        self.scalar = StandardScaler().fit(self.input)
        X_ = self.scalar.transform(self.input)  # transform features
        Y_ = self.target  # target labels remain unchanged
        return X_, Y_

    def compute_gradient(self, X, Y):
        # calculate gradient for a single data point
        hinge_distance = 1 - (Y * np.dot(X, self.weights))
        # if point is within margin or misclassified, gradient differs
        if hinge_distance <= 0:
            gradient = self.weights  # no hinge loss, just regularization term
        else:
            gradient = self.weights - (self.C * Y * X)  # hinge loss contributes
        return gradient

    def compute_gradient_batch(self, X_batch, Y_batch):
        # gradient computation for a batch of samples
        distances = 1 - (Y_batch * np.dot(X_batch, self.weights))
        dw = np.zeros(len(self.weights))  # initialize total gradient sum
        # compute gradient for each point in the batch
        for ind, d in enumerate(distances):
            if d <= 0:
                di = self.weights  # no contribution from hinge loss
            else:
                di = self.weights - (self.C * Y_batch[ind] * X_batch[ind])  # hinge loss gradient
            dw += di
        dw /= len(Y_batch)  # average gradient over the batch
        return dw

    def compute_loss(self, X, Y):
        # calculate the overall loss (regularization + hinge loss)
        reg_loss = 0.5 * np.dot(self.weights, self.weights)  # L2 regularization term
        distances = 1 - Y.flatten() * np.dot(X, self.weights)  # calculate hinge distances
        hinge_loss = self.C * np.sum(np.maximum(0, distances))  # hinge loss (max(0, margin violation))
        loss = reg_loss + hinge_loss  # total loss
        return loss
   
    def stochastic_gradient_descent(self, X, Y, X_val=None, Y_val=None):
        # variables to keep track of progress
        samples = 0
        loss_threshold = self.threshold  # stop training if loss change is too small
        prev_loss = None
        early_stopping_epoch = None
        training_losses = []  # list to store training losses over epochs
        validation_losses = []  # list to store validation losses
        epochs_list = []  # list to keep track of epoch numbers

        for epoch in range(1, self.epoch + 1):
            # shuffle the data to introduce randomness in each epoch
            features, output = shuffle(X, Y)

            for i, feature in enumerate(features):
                # calculate the gradient for the current sample
                gradient = self.compute_gradient(feature, output[i])
                # update weights based on the computed gradient
                self.weights -= self.learning_rate * gradient
                samples += 1

            # calculate the loss after completing one epoch
            train_loss = self.compute_loss(X, Y)
            val_loss = self.compute_loss(X_val, Y_val) if X_val is not None and Y_val is not None else None

            # record losses for visualization purposes
            if epoch % max(1, (self.epoch // 10)) == 0 or epoch == 1:
                training_losses.append(train_loss)
                if val_loss is not None:
                    validation_losses.append(val_loss)
                epochs_list.append(epoch)

            # check for early stopping if the change in loss is below threshold
            if prev_loss is not None:
                loss_difference = abs(prev_loss - train_loss)
                if loss_difference < loss_threshold and early_stopping_epoch is None:
                    early_stopping_epoch = epoch
                    self.weights_at_early_stop = self.weights.copy()
                    print(f"Early stopping at epoch {early_stopping_epoch}")
                    print(f"Total iterations until early stopping: {samples}")
            prev_loss = train_loss

        # print final results after training is completed
        print("Training completed...")
        if early_stopping_epoch is not None:
            print(f"Optimal performance achieved at epoch {early_stopping_epoch}")
        else:
            print("No early stopping was triggered within the given epochs.")
        print("Final weights:")
        print(self.weights)

        return training_losses, validation_losses, epochs_list

    def mini_batch_gradient_descent(self, X, Y, batch_size, X_val=None, Y_val=None):
        # setting up for mini-batch training
        loss_threshold = self.threshold  # early stop threshold
        prev_loss = None
        early_stopping_epoch = None
        iterations = 0  # total iteration count
        iterations_until_early_stop = None
        training_losses = []  # list to track training losses over epochs
        validation_losses = []  # list to track validation losses
        epochs_list = []  # list to keep track of epochs
        num_samples = X.shape[0]  # number of data samples

        for epoch in range(1, self.epoch + 1):
            # shuffle data to ensure randomness each epoch
            features, output = shuffle(X, Y)
            batch_losses = []  # track batch-wise losses

            for i in range(0, num_samples, batch_size):
                # grab a mini-batch of data
                X_batch = features[i:i + batch_size]
                Y_batch = output[i:i + batch_size].flatten()
                # calculate gradient for current batch
                gradient = self.compute_gradient_batch(X_batch, Y_batch)
                # update weights based on gradient
                self.weights -= self.learning_rate * gradient
                iterations += 1

            # calculate loss after the epoch ends
            train_loss = self.compute_loss(X, Y)
            val_loss = self.compute_loss(X_val, Y_val) if X_val is not None and Y_val is not None else None

            # record loss data for visualization
            if epoch % max(1, (self.epoch // 10)) == 0 or epoch == 1:
                training_losses.append(train_loss)
                if val_loss is not None:
                    validation_losses.append(val_loss)
                epochs_list.append(epoch)

            # early stopping check based on small loss changes
            if prev_loss is not None:
                loss_difference = abs(prev_loss - train_loss)
                if loss_difference < loss_threshold and early_stopping_epoch is None:
                    early_stopping_epoch = epoch
                    iterations_until_early_stop = iterations
                    self.weights_at_early_stop = self.weights.copy()
                    print(f"Early stopping at epoch {early_stopping_epoch}")
                    print(f"Iterations until early stopping: {iterations_until_early_stop}")
            prev_loss = train_loss

        # print final results after training completes
        if early_stopping_epoch is not None:
            print(f"Early stopping occurred at epoch {early_stopping_epoch}")
        else:
            print(f"No early stopping within {self.epoch} epochs.")

        print("Training completed.")
        print("Final weights:")
        print(self.weights)
        return training_losses, validation_losses, epochs_list

    def sampling_strategy(self, X, Y):
        # calculate hinge losses for each sample
        distances = 1 - Y.flatten() * np.dot(X, self.weights)
        hinge_losses = np.maximum(0, distances)  # hinge loss values

        # find the sample with the smallest hinge loss (least misclassified)
        min_loss_index = np.argmin(hinge_losses)
        # find the sample with the largest hinge loss (most misclassified)
        # max_loss_index = np.argmax(hinge_losses)

        x = X[min_loss_index]
        y = Y[min_loss_index]
        return x, y, min_loss_index

    def predict(self, X_test, Y_test, weights=None, dataset_name="Evaluation Dataset"):
        # predict labels using specified weights
        if weights is not None:
            optimal_weights = weights
        elif hasattr(self, 'weights_at_early_stop') and self.weights_at_early_stop is not None:
            optimal_weights = self.weights_at_early_stop
        else:
            optimal_weights = self.weights

        # generate predictions for the dataset
        predicted_values = [np.sign(np.dot(X_test[i], optimal_weights)) for i in range(X_test.shape[0])]
        # calculate accuracy, precision, and recall
        accuracy = accuracy_score(Y_test, predicted_values)
        print(f"Accuracy on {dataset_name}: {accuracy:.4f}")
        precision = precision_score(Y_test, predicted_values, pos_label=1)
        print(f"Precision on {dataset_name}: {precision:.4f}")
        recall = recall_score(Y_test, predicted_values, pos_label=1)
        print(f"Recall on {dataset_name}: {recall:.4f}")
        return accuracy, precision, recall


def part_1(X_train, y_train):
    # parameters for SGD

    C = 0.09  # regualization 
    learning_rate = 0.001  # learning rate
    epoch = 2200  # epoch num
    threshold = 1e-3  # early stoppage threshold 

    # initilziae the SVM class
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train, threshold=threshold)

    # pre process step 
    X_train_preprocessed, y_train_preprocessed = my_svm.pre_process()

    # splitting data into 80% train, 20% validation stratified
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train_preprocessed, y_train_preprocessed, test_size=0.2, random_state=42, stratify=y_train_preprocessed
    )

    # train model w SGD
    training_losses, validation_losses, epochs_list = my_svm.stochastic_gradient_descent(
        X_train_new, y_train_new, X_val, y_val
    )

    # plot the  training and validation losses over the epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, training_losses, label='Training Loss -- SGD')
    if validation_losses:  # are  validation losses are available if so do
        plt.plot(epochs_list, validation_losses, label='Validation Loss -- SGD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss -- SGD')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot performance metrics on the validation set
    print("\nFinal Performance on Validation Set:")
    optimal_weights = my_svm.weights_at_early_stop if my_svm.weights_at_early_stop is not None else my_svm.weights
    accuracy_sgd, precision_sgd, recall_sgd = my_svm.predict(X_val, y_val, weights=optimal_weights)
    print(f"Accuracy: {accuracy_sgd}")
    print(f"Precision: {precision_sgd}")
    print(f"Recall: {recall_sgd}")

    return my_svm, training_losses, validation_losses, epochs_list, accuracy_sgd, precision_sgd, recall_sgd

def part_2(X_train, y_train, sgd_training_losses, sgd_validation_losses, sgd_epochs_list, accuracy_sgd, precision_sgd, recall_sgd):
    #  parameters for Mini-Batch Gradient Descent
    batch_size = 12  # size of batch
    C = 0.09 # regualization parameter
    learning_rate = 0.001  # learning rate 
    epoch = 2200  # epoch num
    threshold = 1e-3  #  stopping threshold for loss difference

    # initizalize the class
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train, threshold=threshold)

    #pre process data
    X_train_preprocessed, y_train_preprocessed = my_svm.pre_process()

    # split into 80% 20% split
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train_preprocessed, y_train_preprocessed, test_size=0.2, random_state=42, stratify=y_train_preprocessed
    )

    # train using Mini-Batch Gradient Descent
    mb_training_losses, mb_validation_losses, mb_epochs_list = my_svm.mini_batch_gradient_descent(
        X_train_new, y_train_new, batch_size, X_val, y_val
    )

    # plot training and validation losses for Mini-Batch Gradient Descent
    plt.figure(figsize=(10, 6))
    plt.plot(mb_epochs_list, mb_training_losses, label='Training Loss - Mini-Batch GD')
    plt.plot(mb_epochs_list, mb_validation_losses, label='Validation Loss - Mini-Batch GD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Mini-Batch Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

    # calculate performance metrics on the validation set for Mini-Batch GD
    print("\nFinal Performance on Validation Set - Mini-Batch GD:")
    optimal_weights = my_svm.weights_at_early_stop if my_svm.weights_at_early_stop is not None else my_svm.weights
    accuracy_mb, precision_mb, recall_mb = my_svm.predict(X_val, y_val, weights=optimal_weights)
    print(f"Accuracy: {accuracy_mb}")
    print(f"Precision: {precision_mb}")
    print(f"Recall: {recall_mb}")

    # Compare both of SGD and Mini-Batch GD
    print("\nComparison of Performance Metrics:")
    print(f"SGD Accuracy: {accuracy_sgd}, Mini-Batch GD Accuracy: {accuracy_mb}")
    print(f"SGD Precision: {precision_sgd}, Mini-Batch GD Precision: {precision_mb}")
    print(f"SGD Recall: {recall_sgd}, Mini-Batch GD Recall: {recall_mb}")

    return my_svm, mb_training_losses, mb_validation_losses, mb_epochs_list

def part_3(X_train_full, y_train_full, X_test, y_test):
    # setting parameters for active learning
    C = 0.1  # regularization parameter
    learning_rate = 0.005  # learning rate for the optimizer
    epoch = 2000  # number of epochs for training
    n_initial_samples = 10  # starting number of training samples

    # randomly select initial samples for training
    X_initial, X_rest, y_initial, y_rest = train_test_split(
        X_train_full, y_train_full, train_size=n_initial_samples, random_state=42, stratify=y_train_full
    )

    # treat remaining data as unlabeled
    X_unlabeled, X_val, y_unlabeled, y_val = train_test_split(
        X_rest, y_rest, test_size=0.2, random_state=42, stratify=y_rest
    )

    # create an instance of the SVM class for active learning
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_initial, Y=y_initial)

    # initialize lists to track metrics during training
    total_training_losses = []
    total_validation_losses = []
    total_epochs_list = []
    total_samples_used = []
    total_epochs = 0

    # setting maximum iterations for active learning loop
    max_iterations = 50
    performance_threshold = 0.95  # desired accuracy threshold

    for iteration in range(max_iterations):
        print(f"\n--- Active Learning Iteration {iteration + 1} ---")

        # preprocess data
        X_train_preprocessed, y_train_preprocessed = my_svm.pre_process()
        X_val_preprocessed = my_svm.scalar.transform(X_val)

        # initialize metrics for current training run
        training_losses = []
        validation_losses = []
        epochs_list = []
        prev_loss = None
        loss_threshold = 1e-3  # early stopping threshold
        early_stopping_epoch = None

        for epoch_num in range(1, epoch + 1):
            # shuffle training data for better SGD performance
            features, output = shuffle(X_train_preprocessed, y_train_preprocessed)

            for i, feature in enumerate(features):
                gradient = my_svm.compute_gradient(feature, output[i])
                my_svm.weights -= my_svm.learning_rate * gradient

            # calculate losses for training and validation
            train_loss = my_svm.compute_loss(X_train_preprocessed, y_train_preprocessed)
            val_loss = my_svm.compute_loss(X_val_preprocessed, y_val)

            # record losses periodically for plotting
            if (epoch_num >= 10 and epoch_num % max(1, (epoch // 10)) == 0) or epoch_num == 1:
                training_losses.append(train_loss)
                validation_losses.append(val_loss)
                epochs_list.append(total_epochs + epoch_num)

            # check for early stopping based on minimal loss change
            if prev_loss is not None:
                loss_difference = abs(prev_loss - train_loss)
                if loss_difference < loss_threshold and early_stopping_epoch is None:
                    early_stopping_epoch = epoch_num
                    print(f"Early stopping at epoch {epoch_num}")
            prev_loss = train_loss

        total_epochs += epoch_num  # update total epochs count

        # collect metrics across iterations
        total_training_losses.extend(training_losses)
        total_validation_losses.extend(validation_losses)
        total_epochs_list.extend(epochs_list)
        total_samples_used.append(len(X_initial))

        # calculate validation accuracy
        y_val_pred = np.sign(np.dot(X_val_preprocessed, my_svm.weights))
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Iteration {iteration + 1}, Samples used: {len(X_initial)}, Validation Accuracy: {val_accuracy:.4f}")

        # check for stopping criteria
        if val_accuracy >= performance_threshold:
            print("Desired performance achieved.")
            break
        if len(X_unlabeled) == 0:
            print("No more unlabeled samples.")
            break

        # preprocess remaining unlabeled samples
        X_unlabeled_preprocessed = my_svm.scalar.transform(X_unlabeled)

        # select new sample based on sampling strategy
        x_new_sample, y_new_sample, max_loss_index = my_svm.sampling_strategy(
            X_unlabeled_preprocessed, y_unlabeled
        )

        # add selected sample to the training set
        X_initial = np.vstack([X_initial, X_unlabeled[max_loss_index].reshape(1, -1)])
        y_initial = np.vstack([y_initial, y_unlabeled[max_loss_index].reshape(1, )])

        # remove the selected sample from the pool of unlabeled data
        X_unlabeled = np.delete(X_unlabeled, max_loss_index, axis=0)
        y_unlabeled = np.delete(y_unlabeled, max_loss_index, axis=0)

        # update the SVM classifier with the new training data
        my_svm.input = X_initial
        my_svm.target = y_initial

    # plot overall training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(total_epochs_list, total_training_losses, label='Training Loss')
    plt.plot(total_epochs_list, total_validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - Active Learning')
    plt.legend()
    plt.grid(True)
    plt.show()

    # evaluate final performance on the validation set
    print("\nFinal Performance on Validation Set:")
    accuracy_val_final, precision_val_final, recall_val_final = my_svm.predict(
        X_val_preprocessed, y_val, weights=my_svm.weights, dataset_name="Validation Set"
    )

    # evaluate performance on the test set
    X_test_preprocessed = my_svm.scalar.transform(X_test)
    print("\nFinal Performance on Test Set:")
    accuracy_test_final, precision_test_final, recall_test_final = my_svm.predict(
        X_test_preprocessed, y_test, weights=my_svm.weights, dataset_name="Test Set"
    )

    # output total samples used to achieve desired performance
    print(f"\nNumber of samples used to achieve optimal performance: {len(X_initial)}")
    return my_svm


def main():
    # load the data file into the panda frame
    print("Loading dataset...")
    data = pd.read_csv('data1.csv')

    # show the distribution of the  benign and malignant cases in the data file
    print("Class Distribution:")
    label_distribution = data['diagnosis'].value_counts()
    print(label_distribution)

    # class distrubution showingg the mapping to categories
    print("\nClass Distribution (Detailed):")
    category_dict = {'B': 'Benign', 'M': 'Malignant'}
    detailed_distribution = data['diagnosis'].map(category_dict).value_counts()
    print(detailed_distribution)

    # dropping irrelevant columns n adding a bias term
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
    X = data.iloc[:, 1:]  # select the features
    X.insert(loc=len(X.columns), column="bias", value=1)  # buas tern
    X_features = X.to_numpy()

    #  Benign = -1 Malignant = 1
    category_dict = {'B': -1.0, 'M': 1.0}
    Y = np.array([(data['diagnosis']).to_numpy()]).T
    Y_target = np.vectorize(category_dict.get)(Y)

    # splitting data  into training and test sets (stratified)
    print("\nSplitting dataset into train and test sets...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_features, Y_target, test_size=0.2, random_state=42, stratify=Y_target
    )

    # Part I: SGD 
    print("\n==== Running Part I (Stochastic Gradient Descent - SGD) ====")
    result_part1 = part_1(X_train_full, y_train_full)
    my_svm_sgd, training_losses_sgd, validation_losses_sgd, epochs_list_sgd, accuracy_sgd, precision_sgd, recall_sgd = result_part1

    # noramalize the test set
    X_test_norm = my_svm_sgd.scalar.transform(X_test)

    #  print test performance metrics for SGD
    print("\n==== [SGD] Test Set Performance ====")
    accuracy_sgd_test, precision_sgd_test, recall_sgd_test = my_svm_sgd.predict(
        X_test_norm, y_test, weights=my_svm_sgd.weights_at_early_stop, dataset_name="Test Set"
    )

    # Part II: Mini-Batch GD 
    print("\n==== Running Part II (Mini-Batch Gradient Descent - Mini-Batch GD) ====")
    result_part2 = part_2(X_train_full, y_train_full, training_losses_sgd, validation_losses_sgd, epochs_list_sgd, accuracy_sgd, precision_sgd, recall_sgd)
    my_svm_mini_batch, training_losses_mb, validation_losses_mb, epochs_list_mb = result_part2

    # normalizee test set 
    X_test_norm_mb = my_svm_mini_batch.scalar.transform(X_test)

    # print test performance metrics for Mini-Batch GD
    print("\n==== [Mini-Batch GD] Test Set Performance ====")
    accuracy_mb_test, precision_mb_test, recall_mb_test = my_svm_mini_batch.predict(
        X_test_norm_mb, y_test, weights=my_svm_mini_batch.weights_at_early_stop, dataset_name="Test Set"
    )

    # comapring metrics on the test set
    print("\n==== [Comparison of Test Set Performance] ====")
    print(f"{'Metric':<15}{'SGD':<15}{'Mini-Batch GD':<15}")
    print(f"{'Accuracy':<15}{accuracy_sgd_test:<15.4f}{accuracy_mb_test:<15.4f}")
    print(f"{'Precision':<15}{precision_sgd_test:<15.4f}{precision_mb_test:<15.4f}")
    print(f"{'Recall':<15}{recall_sgd_test:<15.4f}{recall_mb_test:<15.4f}")

    # combined plot losses for both SGD and Mini-Batch GD
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list_sgd, training_losses_sgd, label='Training Loss - SGD')
    plt.plot(epochs_list_sgd, validation_losses_sgd, label='Validation Loss - SGD')
    plt.plot(epochs_list_mb, training_losses_mb, label='Training Loss - Mini-Batch GD')
    plt.plot(epochs_list_mb, validation_losses_mb, label='Validation Loss - Mini-Batch GD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses for SGD and Mini-Batch GD')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Part III: Active Learning Strategy
    print("\n==== Running Part III (Active Learning) ====")
    my_svm_active_learning = part_3(X_train_full, y_train_full, X_test, y_test)




if __name__ == "__main__":
    # run main
    main()
