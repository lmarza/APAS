import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel(100) # suppress deprecation messages
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
 
import numpy as np
import pandas as pd

import argparse

from util import Dataset, Explainer

import csv


np.random.seed(0)
tf.random.set_seed(0)



def nn_model(input_shape):
    x_in = Input(shape=(input_shape,))
    x = Dense(20, activation='relu')(x_in)
    x = Dense(10, activation='relu')(x)
    x_out = Dense(1, activation='sigmoid')(x)
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return nn

def train_model(x_train, y_train):

    nn = nn_model(x_train.shape[1])
    nn.summary()
    nn.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    return nn




def main(args):

    # Logs path
    log_path = args.log_path
    df_results = pd.DataFrame(columns=['Delta', 'Original', 'New'])

    # Loading data
    ds_name = args.dataset_name
    ds = Dataset(args.data_path, ds_name)
    
    delta = args.delta
    samples = args.samples

    if ds_name == "german":
        x_train, y_train, x_test, y_test = ds.load_german()
    else:
        x_train, y_train, x_test, y_test = ds.load_data()

    # Train and evaluate model
    if args.train:
        model = train_model(x_train, y_train)

        model.save(f'{args.model_path}nn_{ds_name}.h5', save_format='h5')
        tf.saved_model.save(model, f'{args.model_path}nn_{ds_name}_saved_model.h5') # used for onnx conversion
    else:
        model = tf.keras.models.load_model(f'{args.model_path}nn_{ds_name}.h5')

    # tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs


    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy: ', score[1])

    X = x_test[:args.runs].reshape((args.runs,) + x_test[1].shape)
    shape = X.shape


    row = 0 

    for i in range(shape[0]):
        # Process one CFX at a time
        X_i = X[i]
        X_i = X_i.reshape(1,shape[1])

        pred_org = model(X_i).numpy()
            
        # Generate CF explanation
        e = Explainer(X_i, model, x_test, y_test)
        cfx = e.explain()
        pred_cfx_org = model(cfx).numpy()

        # Store initial weights
        old_weights = {}
        for l in range(1,len(model.layers)):
            old_weights[l] = model.layers[l].get_weights()
            
        # Now test how predicted value changes
        for s in range(samples):
            for l in range(1,len(model.layers)):
                model.layers[l].set_weights(old_weights[l])

                layer_weights = model.layers[l].get_weights()[0]
                layer_biases  = model.layers[l].get_weights()[1]

                weights_perturbation = np.random.uniform(-delta, delta, layer_weights.shape)
                biases_perturbation = np.random.uniform(-delta, delta, layer_biases.shape)
                
                model.layers[l].set_weights([layer_weights+weights_perturbation,layer_biases+biases_perturbation])

            pred_cfx_pert = model(cfx).numpy()
            

            print(f"Original prediction: {pred_org}. Counterfactual prediction: {pred_cfx_org}. New CFX prediction under perturbation: {pred_cfx_pert}.")
            df_results.loc[row] = [delta, pred_cfx_org.item(), pred_cfx_pert.item()]
            row = row+1
    
    df_results.to_csv(f"{log_path}{ds_name}_delta_{delta}_samples_{samples}.csv", index=False)

    # Get summary
    df_results['Difference'] = (df_results['New'] - df_results['Original']).abs()
    grouped = df_results.groupby('Original',sort=False).mean()   

    original = df_results['Original'].unique()
    grouped['Original'] = original

    grouped = grouped[['Delta','New','Original', 'Difference']]
    
    grouped.to_csv(f"{log_path}summary_{ds_name}_delta_{delta}_samples_{samples}.csv", index=False)



if __name__ == "__main__":

    choices_data = ["german", "news", "spam"]

    parser = argparse.ArgumentParser(description='CFX generation script.')
    parser.add_argument('dataset_name', metavar='ds', default=None, help=f'Dataset name. Supported: {choices_data}', choices=choices_data)
    parser.add_argument('data_path', metavar='dp', default=None, help='Path to dataset.')
    parser.add_argument('model_path', metavar='mp', default=None, help='Path where model should be loaded/saved.')
    parser.add_argument('log_path', metavar='lp', default=None, help='Path where logs should be loaded/saved.')
    parser.add_argument('--train', action="store_true", help='Controls whether model is trained anew or loaded. Default: False.')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to evaluate. Default: 1.')
    parser.add_argument('--delta', type=float, default=0.1, help='Magnitude of maximum model shift. Default: 0.1.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs to be evaluated. Default: 1.')




    args = parser.parse_args()

    main(args)