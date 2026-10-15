#Imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob, torch, torch.utils, librosa, cv2, time
from torch.utils.data import DataLoader
import numpy as np, sounddevice as sd, soundfile as sf
from torchvision import models 

#Set-ups & hyperparameters
base_dir = "C:/Users/maria/Desktop/AI lab"
training_dir = os.path.join(base_dir, "Dataset/Training")
validation_dir = os.path.join(base_dir, "Dataset/Validation")
final_model = "covid_model.pth"
sample_rate = 22050 
num_mels = 128 
img_size = 128 
size_of_batch = 8 
epochs_num = 20  
learning_rate = 0.0005 


#Live recording
def live_audio(duration=5, output_path="live_input_cough.wav"):
    print(f"Recording for {duration} seconds")

    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")

    for sec in range(duration, 0, -1):
        print(f"{sec}s remaining", end="\r", flush=True)
        time.sleep(1)

    sd.wait()
    print("Recording done.")
    sf.write(output_path, recording, sample_rate)
    return output_path


#Audio -> mel-spectrogram -> mel-db
def spectrogram_db(audio_path):
    y, sr = librosa.load(audio_path, sr=sample_rate)

    #Padding/trimming
    target_len = sample_rate * 5
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len] 

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels, fmax=8000)

    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db


#Augmentation (only for training set)
def augmentation(mel):
    mel = mel.copy()
    n_mels, time_steps = mel.shape

    #Frequency masking
    f_mask = np.random.randint(0, 20)
    f_start = np.random.randint(0, max(1, n_mels - f_mask+1))
    mel[f_start:f_start + f_mask, :] = mel.min()

    #Time masking 
    t_mask = np.random.randint(0, 20)
    t_start = np.random.randint(0, max(1, time_steps - t_mask+1))
    mel[:, t_start:t_start + t_mask] = mel.min()

    return mel


#Dataset loading, data augmentation, normalization, optimization
class LoadingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, augment=False):
        self.file_list = []
        self.labels = []
        self.augment = augment

        class_map = {"healthy": 0, "covid_positive": 1}

        for category, label in class_map.items():
            folder = os.path.join(dataset_root, category)

            for ext in ["*.wav", "*.mp3", "*.m4a"]:
                for file_path in glob.glob(os.path.join(folder, ext)):
                    self.file_list.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        features = spectrogram_db(audio_path)

        features = cv2.resize(features, (img_size, img_size))

        if self.augment:
            features = augmentation(features)

        features = (features - features.min()) / (features.max() - features.min() + 1e-8)

        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return features, label


#ResNet18
def resnet_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    original_conv = model.conv1
    model.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    
    with torch.no_grad():
        model.conv1.weight = torch.nn.Parameter(
            original_conv.weight.mean(dim=1, keepdim=True)
        )

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(model.fc.in_features, 2)
    )

    return model


#Training the model 
def training_model():
    print("\nStarting training pipeline.\n")

    train_set = LoadingDataset(training_dir, augment=True)
    val_set = LoadingDataset(validation_dir, augment=False)

    train_size = len(train_set)
    val_size = len(val_set)
    print(f"Number of samples: {train_size} training / {val_size} validation\n")

    train_loader = DataLoader(train_set, batch_size=size_of_batch, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_set,   batch_size=size_of_batch, shuffle=False, num_workers=0)

    labels_all = train_set.labels
    class_counts = np.bincount(labels_all, minlength=2)
 
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)

    model = resnet_model()
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="max", factor=0.5, patience=5 )

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs_num + 1):

        #Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_acc = 100 * train_correct / train_total

        #Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr}")

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(model.state_dict(), final_model)
            marker = " [Best saved]"

        print(f"Epoch {epoch:02d}/{epochs_num}  |  "
              f"Train loss: {train_loss/len(train_loader):.4f}  acc: {train_acc:.1f}%  |  "
              f"Val loss: {val_loss/max(1,len(val_loader)):.4f}  acc: {val_acc:.1f}%"
              f"{marker}")


#Input and Predict
def classify_audio(audio_path):
    mel_db = spectrogram_db(audio_path)

    features = cv2.resize(mel_db, (img_size, img_size))
    lo, hi = features.min(), features.max()
    features = (features - lo) / (hi - lo + 1e-8)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    model = resnet_model()
    model.load_state_dict(torch.load(final_model))
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item() * 100

    labels = {0: "Healthy", 1: "COVID-19 Positive"}
    print(f"\nResult: {labels[pred]}")
    print(f"Confidence: {conf:.1f}%")


#Running
if __name__ == "__main__":

    training = False    
    input_audio = True 
    file_audio = False

    if training:
        training_model()

    elif input_audio:
        wav = live_audio(duration=5, output_path="temp_live_mic.wav")
        if wav and os.path.exists(wav):
            classify_audio(wav)

    #Example usage   
    elif file_audio:
        target = "C://Users/maria/Downloads/506b5606-cecb-4fba-8723-c4a2bfb8b038.mp3"
        classify_audio(target)