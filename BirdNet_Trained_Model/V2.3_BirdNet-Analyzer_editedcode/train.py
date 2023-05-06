import os
import sys
import argparse

import numpy as np

import config as cfg
import audio
import model

####//////////ALICE ///////////////#################

def loadNYData():
    
    # Because the subfolder names are used as labels either I need to bucket the data into folders OR I use the .csv to generate the labels
    
    #Step 1: Define Path to Data
    #Step 1.1: Load df
    #Step 2: Randomly Sample data into x_train and y_train buckets
    #Step 3: Generate Labels
    #Step 4: Concatenate path to my directory with ID of the file
    
    df = pd.read_csv(f'{cfg.TRAIN_DATA_PATH}/ny_metadata.csv')
    
    # Get list of files
    files = df.id.apply(lambda x:f'{cfg.TRAIN_DATA_PATH}/data/{x}.wav').values  # Values provides a list back of the files
    
    # Load training data
    x_train = []
    y_train = pd.get_dummies(df.sp,columns=['sp']).values
    
    
    #x_train, x_, y_train, y_ = train_test_split(df.id, df.sp, test_size=0.20, random_state=0) Dont need to split data becuase

    labels = df.sp.unique()

 
            # Load files
    for f in files:
        # Load audio
        sig, rate = audio.openAudioFile(f)
        
        # Crop center segment
        sig = audio.cropCenter(sig, rate, cfg.SIG_LENGTH)

        # Get feature embeddings
        embeddings = model.embeddings([sig])[0]

        # Add to training data
        x_train.append(embeddings)

    # Convert to numpy arrays
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')

    return x_train, y_train, labels 

#############////////// ///////////////#################

def loadTrainingData():

    # Get list of subfolders as labels
    labels = [l for l in sorted(os.listdir(cfg.TRAIN_DATA_PATH))]
    
    
    print(labels)

    # Load training data
    x_train = []
    y_train = []
    for i, label in enumerate(labels):
            
            # Get label vector            
            label_vector = np.zeros((len(labels),), dtype='float32')
            if not label.lower() in ['noise', 'other', 'background', 'silence']:
                label_vector[i] = 1
    
            # Get list of files
            files = [os.path.join(cfg.TRAIN_DATA_PATH, label, f) for f in sorted(os.listdir(os.path.join(cfg.TRAIN_DATA_PATH, label))) if f.rsplit('.', 1)[1].lower() in ['wav', 'flac', 'mp3', 'ogg', 'm4a']]
            
            print(files)
            
    
            # Load files
            for f in files:
    
                # Load audio
                sig, rate = audio.openAudioFile(f)
    
                # Crop center segment
                sig = audio.cropCenter(sig, rate, cfg.SIG_LENGTH)

                # Get feature embeddings
                embeddings = model.embeddings([sig])[0]

                # Add to training data
                x_train.append(embeddings)
                y_train.append(label_vector)

    # Convert to numpy arrays
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')

    return x_train, y_train, labels

def trainModel():

    # Load training data
    print('Loading training data...', flush=True)
    x_train, y_train, labels = loadTrainingData()
    print('...Done. Loaded {} training samples and {} labels.'.format(x_train.shape[0], y_train.shape[1]), flush=True)

    # Build model
    print('Building model...', flush=True)    
    classifier = model.buildLinearClassifier(y_train.shape[1], x_train.shape[1])
    print('...Done.', flush=True)

    # Train model
    print('Training model...', flush=True)
    classifier, prec, fbeta_score = model.trainLinearClassifier(classifier,
                                                       x_train, 
                                                       y_train, 
                                                       epochs=cfg.TRAIN_EPOCHS,
                                                       batch_size=cfg.TRAIN_BATCH_SIZE,
                                                       learning_rate=cfg.TRAIN_LEARNING_RATE)
    model.saveLinearClassifier(classifier, cfg.CUSTOM_CLASSIFIER, labels)
    print('...Done. Best top-1 precision: {}'.format(prec), flush=True)
    print('...Done. FBetaScore: {}'.format(fbeta_score), flush=True)
    #print('...Done. MAPScore: {}'.format(mean_average_precision), flush=True)

if __name__ == '__main__':

    # Clear error log
    #clearErrorLog()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze audio files with BirdNET')
    parser.add_argument('--i', default='train_data/', help='Path to training data folder. Subfolder names are used as labels.')
    parser.add_argument('--o', default='checkpoints/custom/Custom_Classifier.tflite', help='Path to trained classifier model output.')
    parser.add_argument('--epochs', type=int,default=100, help='Number of training epochs. Defaults to 100.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. Defaults to 32.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate. Defaults to 0.01.')
    parser.add_argument('--hidden_units', type=int, default=0, help='Number of hidden units. Defaults to 0. If set to >0, a two-layer classifier is used.')

    args = parser.parse_args()

    # Config
    cfg.TRAIN_DATA_PATH = args.i
    cfg.CUSTOM_CLASSIFIER = args.o
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.TRAIN_LEARNING_RATE = args.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = args.hidden_units

    # Train model
    trainModel()
