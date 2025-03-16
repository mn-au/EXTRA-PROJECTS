# Import base libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import Scikit-learn libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms

# set the random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# configure the device being used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

######################################################################################

# Part I: Fashion-MNIST Classification

# configuration variables for Part I
BATCH_SIZE = 64          # batch size for loading data
LEARNING_RATE = 0.01     # learning rate for the optimizer
NUM_EPOCHS = 30        # number of epochs for training the model
NUM_WORKERS = 2          # number of worker threads for data loading

class FashionMNISTDataLoader:
    """
    class to load and preprocess the Fashion-MNIST dataset.
    """
    def __init__(self, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        # initialize the batch size and number of workers
        self.batch_size = batch_size
        self.num_workers = num_workers

        # placeholders for data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self):
        # define transformations to apply to the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert images to tensors
            # include normalization if needed
            # transforms.Normalize((0.5,), (0.5,))
        ])

        # load the Fashion-MNIST training and test datasets
        try:
            train_dataset = datasets.FashionMNIST(
                root='data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(
                root='data', train=False, download=True, transform=transform)
        except Exception as e:
            print(f"Error downloading the dataset: {e}")
            return

        # split the training dataset into training and validation sets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])

        # print dataset sizes for verification
        print(f"Train set size: {train_size}, Validation set size: {val_size}, Test set size: {len(test_dataset)}")

        # create data loaders for training, validation, and testing
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class FashionCNNModel(nn.Module):
    """
    convolutional neural network model for Fashion-MNIST classification.
    """
    def __init__(self):
        super(FashionCNNModel, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)  # first convolutional layer
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3)  # second convolutional layer
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3)  # third convolutional layer

        # define fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # first fully connected layer
        self.fc2 = nn.Linear(120, 84)         # second fully connected layer
        self.fc3 = nn.Linear(84, 10)          # output layer (10 classes)

    def forward(self, x):
        # apply convolutional layers with ReLU activation and pooling
        x = F.relu(self.conv1(x))  # apply first convolutional layer
        x = F.max_pool2d(x, 2)     # apply max pooling
        x = F.relu(self.conv2(x))  # apply second convolutional layer
        x = F.relu(self.conv3(x))  # apply third convolutional layer
        x = F.max_pool2d(x, 2)     # apply max pooling

        # flatten the tensor for fully connected layers
        x = x.view(-1, 16 * 4 * 4)

        # apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))  # apply first fully connected layer
        x = F.relu(self.fc2(x))  # apply second fully connected layer
        x = self.fc3(x)          # output layer without activation
        return x

class FashionMNISTTrainer:
    """
    class to train the FashionCNN model.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
        # initialize model, data loaders, criterion, optimizer, and number of epochs
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        # placeholders for storing losses
        self.train_losses = []
        self.val_losses = []

    def train(self):
        # training loop for the specified number of epochs
        for epoch in range(self.num_epochs):
            self.model.train()  # set the model to training mode
            running_loss = 0.0  # variable to accumulate loss

            # iterate over batches of training data
            for images, labels in self.train_loader:
                images = images.to(device)  # move images to the configured device
                labels = labels.to(device)  # move labels to the configured device

                # clear gradients from previous iterations
                self.optimizer.zero_grad()

                # forward pass through the model
                outputs = self.model(images)

                # compute the loss
                loss = self.criterion(outputs, labels)

                # backward pass to compute gradients
                loss.backward()

                # update the model weights
                self.optimizer.step()

                # accumulate batch loss
                running_loss += loss.item() * images.size(0)

            # calculate average training loss for the epoch
            epoch_train_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(epoch_train_loss)

            # validate the model
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # print training and validation loss for the current epoch
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        return self.train_losses, self.val_losses

    def validate(self):
        # validation loop to compute loss on the validation dataset
        self.model.eval()  # set the model to evaluation mode
        val_running_loss = 0.0  # variable to accumulate loss

        with torch.no_grad():
            # iterate over batches of validation data
            for images, labels in self.val_loader:
                images = images.to(device)  # move images to the configured device
                labels = labels.to(device)  # move labels to the configured device

                # forward pass through the model
                outputs = self.model(images)

                # compute the loss
                loss = self.criterion(outputs, labels)

                # accumulate batch loss
                val_running_loss += loss.item() * images.size(0)

        # calculate average validation loss
        epoch_val_loss = val_running_loss / len(self.val_loader.dataset)
        return epoch_val_loss

class FashionMNISTEvaluator:
    """
    class to evaluate the trained FashionCNN model.
    """
    def __init__(self, model, test_loader, class_names):
        # initialize the model, test data loader, and class names
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names

    def evaluate(self):
        # evaluate the model on the test set
        self.model.eval()  # set the model to evaluation mode
        correct = 0  # counter for correctly classified examples
        total = 0  # counter for total examples
        class_correct = [0] * 10  # counters for correctly classified examples per class
        class_total = [0] * 10    # counters for total examples per class

        with torch.no_grad():
            # iterate over batches of test data
            for images, labels in self.test_loader:
                images = images.to(device)  # move images to the configured device
                labels = labels.to(device)  # move labels to the configured device

                # forward pass through the model
                outputs = self.model(images)

                # get the predicted class
                _, predicted = torch.max(outputs, 1)

                # update counters
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # update per-class counters
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == labels[i]).item()
                    class_total[label] += 1

        # calculate overall test accuracy
        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        # calculate and print accuracy for each class
        print("\nPer-Class Accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print(f'Accuracy of {self.class_names[i]}: {accuracy:.2f}%')
            else:
                print(f'No samples for class {self.class_names[i]}')

        return test_accuracy

class FashionMNISTPlotter:
    """
    class to plot training and validation losses.
    """
    @staticmethod
    def plot_losses(train_losses, val_losses):
        # plot the training and validation losses over epochs
        plt.figure(figsize=(10, 5))  # set the figure size
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')  # plot training losses
        plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')  # plot validation losses
        plt.xlabel('Epoch')  # label for the x-axis
        plt.ylabel('Loss')  # label for the y-axis
        plt.title('Fashion-MNIST Training and Validation Loss')  # plot title
        plt.legend()  # add legend to differentiate lines
        plt.show()  # display the plot

def main_part1():
    # part I: Fashion-MNIST
    print("\n===== Part I: Fashion-MNIST =====\n")

    # initialize data loader and load data
    data_loader = FashionMNISTDataLoader()
    data_loader.load_data()

    # define class names for Fashion-MNIST
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    ]

    # instantiate the model
    fashion_model = FashionCNNModel().to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # cross-entropy loss for multi-class classification
    optimizer = torch.optim.SGD(fashion_model.parameters(), lr=LEARNING_RATE)  # stochastic gradient descent optimizer

    # train the model
    trainer = FashionMNISTTrainer(
        model=fashion_model,
        train_loader=data_loader.train_loader,
        val_loader=data_loader.val_loader,
        criterion=criterion,
        optimizer=optimizer
    )
    train_losses, val_losses = trainer.train()

    # plot losses
    plotter = FashionMNISTPlotter()
    plotter.plot_losses(train_losses, val_losses)

    # evaluate the model
    evaluator = FashionMNISTEvaluator(
        model=fashion_model,
        test_loader=data_loader.test_loader,
        class_names=class_names
    )
    test_accuracy = evaluator.evaluate()

######################################################################################

# Part II: COMPAS Bias Measurement and Mitigation

# configuration variables for Part II
LR_LEARNING_RATE = 0.01  # learning rate for logistic regression
LR_NUM_EPOCHS = 1000     # number of epochs for logistic regression
LR_BATCH_SIZE = 64       # batch size for logistic regression

def load_compas_data():
    """
    load and preprocess the COMPAS dataset.
    """
    try:
        # initialize the data processor with the dataset filepath
        data_processor = DataProcessor('compas-scores.csv')
        data_processor.prepare_data()  # prepare the data for processing
        return data_processor
    except FileNotFoundError as e:
        # handle the case where the dataset file is missing
        print("Error: The file 'compas-scores.csv' was not found. Please ensure the file is in the correct location.")
        raise e
    except Exception as e:
        # handle other unexpected errors
        print(f"An unexpected error occurred while loading or processing the dataset: {e}")
        raise e

class DataProcessor:
    """
    class for processing and preparing the COMPAS dataset.
    """
    def __init__(self, filepath):
        # initialize the file path and other variables for data processing
        self.filepath = filepath
        self.data = None
        self.categorical_features = None
        self.numerical_features = None
        self.target_variable = 'reoffend'
        self.sensitive_attribute = 'race'
        self.preprocessor = None
        self.train_data = None
        self.test_data = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        # load the dataset from the specified file
        self.data = pd.read_csv(self.filepath)
        print("Data loaded successfully.")

    def assign_labels(self):
        # assign binary labels for recidivism based on the 'score_text' column
        self.data['reoffend'] = self.data['score_text'].apply(
            lambda x: 0 if x in ['Low', 'Medium'] else (1 if x == 'High' else None))
        initial_shape = self.data.shape
        # remove rows with undefined labels
        self.data.dropna(subset=['reoffend'], inplace=True)
        print(f"Assigned labels. Dropped {initial_shape[0] - self.data.shape[0]} rows with undefined labels.")

    def drop_irrelevant_columns(self):
        # drop columns that are not relevant to the analysis
        self.data = self.data.drop(columns=[
            'id', 'name', 'first', 'last', 'c_case_number', 'c_arrest_date',
            'r_case_number', 'vr_case_number', 'num_r_cases', 'num_vr_cases',
            'vr_charge_desc', 'screening_date', 'score_text'
        ])
        print("Dropped irrelevant columns.")

    def process_time_features(self):
        # process date columns and compute additional time-related features
        date_columns = [
            'dob', 'compas_screening_date', 'c_offense_date', 
            'c_jail_in', 'c_jail_out', 'r_jail_in', 'r_jail_out', 'vr_offense_date'
        ]
        for col in date_columns:
            # convert date columns to datetime format
            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')

        # compute various time-related features
        self.data['age_at_screening'] = (self.data['compas_screening_date'] - self.data['dob']).dt.days / 365.25
        self.data['length_of_incarceration'] = (self.data['c_jail_out'] - self.data['c_jail_in']).dt.days
        self.data['time_since_last_offense'] = (self.data['compas_screening_date'] - self.data['c_offense_date']).dt.days
        self.data['time_since_recent_jail_release'] = self.data[['c_jail_out', 'r_jail_out']].max(axis=1)
        self.data['time_since_recent_jail_release'] = (
            self.data['compas_screening_date'] - self.data['time_since_recent_jail_release']
        ).dt.days
        self.data['length_of_previous_incarceration'] = (self.data['r_jail_out'] - self.data['r_jail_in']).dt.days
        self.data['time_since_violent_offense'] = (self.data['compas_screening_date'] - self.data['vr_offense_date']).dt.days

        # rely on preprocessing pipeline for missing value handling
        print("Processed time features (without manual imputation).")

    def separate_features(self):
        # separate categorical and numerical features
        X = self.data.drop(columns=[self.target_variable])
        self.categorical_features = X.select_dtypes(include=['object']).columns
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        print("Separated categorical and numerical features.")

    def setup_preprocessor(self):
        # set up preprocessing pipelines for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # combine preprocessing pipelines into a column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        print("Preprocessor set up.")

    def split_data(self):
        # split the data into training and testing sets
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]
        self.train_data, self.test_data, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        print("Data split into training and testing sets.")

    def prepare_data(self):
        # Execute all preprocessing steps in sequence
        self.load_data()
        self.assign_labels()
        self.drop_irrelevant_columns()
        self.process_time_features()
        self.separate_features()
        self.setup_preprocessor()
        self.split_data()
        
        # Fit the preprocessor on training data
        self.preprocessor.fit(self.train_data)
        
        # Print information about the training and testing splits
        print("\nTraining Data Distribution by Race:")
        print(self.train_data[self.sensitive_attribute].value_counts())

        print("\nTesting Data Distribution by Race:")
        print(self.test_data[self.sensitive_attribute].value_counts())

        print("Data prepared and preprocessor fitted.")

class LogisticRegressionModel(nn.Module):
    """
    logistic regression model using PyTorch.
    """
    def __init__(self, input_dim):
        # initialize the linear layer with a single output
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # compute the sigmoid activation on the linear output
        return torch.sigmoid(self.linear(x))

class ModelTrainer:
    """
    class for training and evaluating the logistic regression model.
    """
    def __init__(self, input_dim, learning_rate=LR_LEARNING_RATE, num_epochs=LR_NUM_EPOCHS, batch_size=LR_BATCH_SIZE):
        # initialize the logistic regression model, loss function, optimizer, and training parameters
        self.model = LogisticRegressionModel(input_dim).to(device)
        self.criterion = nn.BCELoss()  # binary cross-entropy loss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)  # stochastic gradient descent
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self, X_train, y_train):
        # loop over the specified number of epochs to train the model
        for epoch in range(self.num_epochs):
            self.model.train()  # set the model to training mode
            outputs = self.model(X_train)  # compute predictions
            loss = self.criterion(outputs, y_train)  # calculate loss

            self.optimizer.zero_grad()  # reset gradients
            loss.backward()  # compute gradients
            self.optimizer.step()  # update model parameters

            # log progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, X_test, y_test):
        # evaluate the model on the test dataset
        self.model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(X_test)  # compute predictions
            predicted = outputs.round()  # round predictions to binary values
            correct = (predicted == y_test).sum().item()  # count correct predictions
            accuracy = 100 * correct / y_test.size(0)  # calculate accuracy percentage
        return accuracy, predicted.cpu().numpy()  # return accuracy and predictions

class BiasAnalyzer:
    """
    class for computing bias metrics and balancing the dataset.
    """
    def compute_equalized_odds(self, y_true, y_pred, races):
        metrics = {}
        unique_races = np.unique(races)  # get unique racial groups

        # iterate over racial groups to calculate TPR and FPR
        for group in unique_races:
            group_mask = (races == group)  # mask for the current group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            tp = np.sum((y_pred_group == 1) & (y_true_group == 1))  # true positives
            fn = np.sum((y_pred_group == 0) & (y_true_group == 1))  # false negatives
            fp = np.sum((y_pred_group == 1) & (y_true_group == 0))  # false positives
            tn = np.sum((y_pred_group == 0) & (y_true_group == 0))  # true negatives

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # true positive rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # false positive rate

            # store metrics for the current group
            metrics[group] = {'TPR': tpr, 'FPR': fpr}

            # print TPR and FPR for the group
            print(f"Race Group {group}:")
            print(f"  TPR = {tpr:.4f}, FPR = {fpr:.4f}")

        # calculate differences in TPR and FPR across groups
        tprs = [metrics[group]['TPR'] for group in unique_races]
        fprs = [metrics[group]['FPR'] for group in unique_races]

        tpr_diff = max(tprs) - min(tprs)  # difference in TPR
        fpr_diff = max(fprs) - min(fprs)  # difference in FPR

        # store and print the Equalized Odds Difference
        metrics['Equalized_Odds_Difference'] = max(tpr_diff, fpr_diff)
        print(f"\nEqualized Odds Difference: {metrics['Equalized_Odds_Difference']:.4f}")

        return metrics

    def balance_dataset(self, X_train_df, y_train, sensitive_attribute):
        # create a copy of the training data for manipulation
        train_df = X_train_df.copy()
        train_df['class'] = y_train.values

        resampled_train = []  # list to store resampled data

        # iterate over unique racial groups
        for race in train_df[sensitive_attribute].unique():
            # filter data for the current racial group
            race_subset = train_df[train_df[sensitive_attribute] == race]

            # separate classes within the group
            class_0 = race_subset[race_subset['class'] == 0]
            class_1 = race_subset[race_subset['class'] == 1]

            # determine target size for balancing
            target_size = min(len(class_0), len(class_1))

            # skip balancing if a class is absent
            if target_size == 0:
                continue

            # resample both classes to the target size
            class_0_resampled = resample(class_0, replace=False, n_samples=target_size, random_state=42)
            class_1_resampled = resample(class_1, replace=False, n_samples=target_size, random_state=42)

            # add resampled data to the list
            resampled_train.append(class_0_resampled)
            resampled_train.append(class_1_resampled)

        # concatenate all resampled data and shuffle
        balanced_train_df = pd.concat(resampled_train).sample(frac=1, random_state=42).reset_index(drop=True)

        # check if the resulting dataset is too small
        if balanced_train_df.shape[0] < len(X_train_df) * 0.1:
            raise ValueError("The balanced dataset is too small. Consider revising the balancing strategy.")

        # split features and labels from the balanced data
        X_train_balanced = balanced_train_df.drop(columns=['class'])
        y_train_balanced = balanced_train_df['class']

        print("\nTraining data balanced across races and classes.")

        return X_train_balanced, y_train_balanced

    def count_predictions_by_group(self, y_true, y_pred, races):
        counts = {}
        unique_races = np.unique(races)

        for group in unique_races:
            group_mask = (races == group)
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # True Positives: Predicted reoffend (1) and actually reoffend (1)
            true_positives = np.sum((y_pred_group == 1) & (y_true_group == 1))

            # False Positives: Predicted reoffend (1) but actually no reoffend (0)
            false_positives = np.sum((y_pred_group == 1) & (y_true_group == 0))

            counts[group] = {'True Positives': true_positives, 'False Positives': false_positives}

        return counts

def main_part2():
    print("\n===== Part II: COMPAS Bias Measurement and Mitigation =====\n")

    # Load and preprocess data
    data_processor = load_compas_data()

    # Extract training and testing data
    X_train_df = data_processor.train_data.copy()
    y_train = data_processor.y_train.copy()
    X_test_df = data_processor.test_data.copy()
    y_test = data_processor.y_test.copy()

    # Print examples of training and testing samples
    print("\nSample Training Data:")
    print(X_train_df.head())

    print("\nSample Testing Data:")
    print(X_test_df.head())

    # Define sensitive attribute and initialize bias analyzer
    sensitive_attribute = data_processor.sensitive_attribute
    bias_analyzer = BiasAnalyzer()

    # Train biased and unbiased models
    for bias in [False, True]:
        if not bias:
            print("\n\nTraining Biased Model (Before Bias Mitigation):\n")
            
            # Transform training and test data
            X_train_processed = data_processor.preprocessor.transform(X_train_df)
            X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        else:
            print("\n\nTraining Unbiased Model (After Bias Mitigation):\n")
            
            # Balance the training data to mitigate bias
            X_train_balanced, y_train_balanced = bias_analyzer.balance_dataset(
                X_train_df, y_train, sensitive_attribute
            )
            
            # Print the distribution of the balanced dataset
            print("\nBalanced Training Data Distribution by Race:")
            print(X_train_balanced[sensitive_attribute].value_counts())

            # Transform the balanced training data
            X_train_processed = data_processor.preprocessor.transform(X_train_balanced)
            X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_balanced.values, dtype=torch.float32).view(-1, 1)

        # Prepare test data (shared for both biased and unbiased models)
        X_test_processed = data_processor.preprocessor.transform(X_test_df)
        X_test_tensor = torch.tensor(X_test_processed.toarray(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Dynamically define input dimension based on training data
        input_dim = X_train_tensor.shape[1]

        # Train the logistic regression model
        model_trainer = ModelTrainer(input_dim)
        model_trainer.train(X_train_tensor, y_train_tensor)

        # Evaluate the model on the test set
        accuracy, predicted = model_trainer.evaluate(X_test_tensor, y_test_tensor)
        print(f'\nTest Accuracy: {accuracy:.2f}%')

        # Analyze and print bias metrics using Equalized Odds
        metrics = bias_analyzer.compute_equalized_odds(
            y_true=y_test_tensor.numpy().flatten(),
            y_pred=predicted.flatten(),
            races=X_test_df[sensitive_attribute].values
        )

        # Analyze and print prediction counts by racial group
        prediction_counts = bias_analyzer.count_predictions_by_group(
            y_true=y_test_tensor.numpy().flatten(),
            y_pred=predicted.flatten(),
            races=X_test_df[sensitive_attribute].values
        )

        print("\nPrediction Counts by Race Group:")
        for group, counts in prediction_counts.items():
            print(f"Race Group {group}: True Positives = {counts['True Positives']}, False Positives = {counts['False Positives']}")

def main():
    # call Part I
    #main_part1()

    # call Part II
    main_part2()

if __name__ == "__main__":
    main()
