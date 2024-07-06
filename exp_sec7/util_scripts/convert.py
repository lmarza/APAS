from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

import numpy as np



def extract_sklearn_params(clf):

    input_size = clf.n_features_in_
    n_layers = clf.n_layers_
    output_size = clf.n_outputs_

    # Renaming needed bc sklearn and tensorflow use different names
    if clf.out_activation_ == "logistic":
        output_act = "sigmoid" 
    else:
        raise ValueError(f'Activation {clf.out_activation_} not supported!')

    h_act = clf.get_params()['activation']
    optimizer = clf.get_params()['solver']

    params = {}
    for i, (w,b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        w_tens = np.array(w)
        b_tens = np.array(b)
        params[i+1] = [w_tens, b_tens]

    return input_size, n_layers, output_size, output_act, h_act, optimizer, params

def custom_nn_model(input_shape, num_layers, layers, output_shape, h_activation, output_activation, solver):
    x_in = Input(shape=(input_shape,))

    for k, v in layers.items():
        if k == 1:
            x = Dense(v[1].shape[0], activation=h_activation)(x_in)
        elif k == num_layers-1:
            x = Dense(output_shape, activation=output_activation)(x)
        else:
            x = Dense(v[1].shape[0], activation=h_activation)(x)
   
    nn = Model(inputs=x_in, outputs=x)
    nn.compile(loss='binary_crossentropy', optimizer=solver, metrics=['accuracy'])
    return nn
