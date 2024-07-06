from alibi.confidence import TrustScore

import numpy as np


# from sklearn.preprocessing import normalize

# from sklearn.neighbors import LocalOutlierFactor

# import dice_ml

import pandas as pd

import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# import time

# import matplotlib.pyplot as plt

# debug = True


class Dataset:

    def __init__(self, path, ds_name):
        self.path = path
        self.ds_name = ds_name


    def load_german(self):

        df_train = pd.read_csv (f'{self.path}train.csv').iloc[: , 1:]
        df_test = pd.read_csv (f'{self.path}test.csv').iloc[: , 1:]

        # Extract feature names and target
        names = list(df_train.columns)
        feature_names = names[:-1]
        target = names[-1]

        # Convert to numpy for later use
        x_train, y_train = df_train[feature_names].to_numpy(), df_train[target].to_numpy()
        # y_train = to_categorical(y_train)

        x_test, y_test = df_test[feature_names].to_numpy(), df_test[target].to_numpy()
        # y_test = to_categorical(y_test)
     
        return x_train, y_train, x_test, y_test 

    def min_max_scale(self, df, continuous, min_vals=None, max_vals=None):
        df_copy = copy.copy(df)
        for i, name in enumerate(continuous):
            if min_vals is None:
                min_val = np.min(df_copy[name])
            else:
                min_val = min_vals[i]
            if max_vals is None:
                max_val = np.max(df_copy[name])
            else:
                max_val = max_vals[i]
            df_copy[name] = (df_copy[name] - min_val) / (max_val - min_val)
        return df_copy

    def load_data(self):
        df = pd.read_csv(f'{self.path}{self.ds_name}.csv')
        df = df.dropna()

        if self.ds_name == "no2":
            df = df.replace(to_replace={'N': 0, 'P': 1})

        ordinal_features = {}
        discrete_features = {}
        continuous_features = list(df.columns)[:-1]

        target = "Outcome"

        # min max scale
        min_vals = np.min(df[continuous_features], axis=0)
        max_vals = np.max(df[continuous_features], axis=0)

        df_mm = self.min_max_scale(df, continuous_features, min_vals, max_vals)
        columns = list(df_mm.columns)
        # get X, y
        X, y = df_mm.drop(columns=['Outcome']), pd.DataFrame(df_mm['Outcome'])

        SPLIT = .2
        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=SPLIT, shuffle=True,
                                                            random_state=0)
        
        x_train, y_train = x_train.to_numpy(), y_train.to_numpy()
        # y_train = to_categorical(y_train)

        x_test, y_test = x_test.to_numpy(), y_test.to_numpy()
        # y_test = to_categorical(y_test)

        return x_train, y_train, x_test, y_test 



class Explainer:

    def __init__(self, x, model, X, Y):
        self.inpt = x
        self.model = model
        self.x_data = X
        self.y_data = Y
        self.kdtrees = self.fit_kdtree(self.x_data, self.y_data)



    def fit_kdtree(self, X, Y):

        y_cat = to_categorical(self.y_data)

        # Fit kdtree using alibi   
        ts = TrustScore()
        ts.fit(self.x_data, y_cat, classes=2)  

        # Return one kdtree for each class
        return ts.kdtrees
        
    def explain(self):

        # Fetch kdtree of counterfactual class
        pred = self.model.predict(self.inpt)
        c = np.where(pred > 0.5, 1, 0)  
        cf_class = 1 - c.item()
        
        # Return closest neighbour that flips class
        dist, idx = self.kdtrees[cf_class].query(self.inpt, k=1)
        idx = idx.flatten()
        tree_data, _, _, _ = self.kdtrees[cf_class].get_arrays()

        cfx = tree_data[idx]
    
        return cfx


# class Stats:
#     def __init__(self, inputs, model, total_cfx):
#         self.inputs = inputs
#         self.model = model 
#         self.ranges = {'min': None, 'max': None}
#         self._compute_ranges()
#         self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
#         self.lof.fit(self.inputs)
#         self.results = {}

#         if total_cfx == 1:
#             self.get_stats_summary = self. get_summary_single
#             self.evaluate = self.evaluate_explanation
#         else:
#             self.get_stats_summary = self. get_summary_set
#             self.evaluate = self.evaluate_explanations

#     def _compute_ranges(self):
#         self.ranges['min'] = self.inputs.min(axis=0)
#         self.ranges['max'] = self.inputs.max(axis=0)


#     def _compute_normalised_l1(self, x, cfx):
#         return np.sum(np.abs(x - cfx)) / (cfx.shape[0])
    
#     def _compute_l2(self, x, cfx):
#         return np.linalg.norm(x - cfx)
    
#     def _compute_distance(self,x,cfx,norm):
#         if norm == 1:
#             return self._compute_normalised_l1(x,cfx)
#         elif norm == 2:
#             return self._compute_l2(x,cfx)
#         else:
#             raise

#     def _check_validity(self, x, cfx):
#         for el in cfx: 
        
#             if (np.argmax(self.model.predict(x), axis=1)[0] == np.argmax(self.model.predict(el.cf['X']), axis=1)[0]):
#                 return False
                
#         return True

#     def _compute_lof(self, cfx):
#         return self.lof.predict(cfx)[0]

#     def _compute_k_distance(self, x, cfx, norm):
#         return sum([self._compute_distance(x, el.cf['X'],norm) for el in cfx])/len(cfx)

#     def _compute_k_diversity(self, cfx, norm):
#         from itertools import combinations
#         comb = set(list(combinations(cfx,2)))

#         if len(comb) == 0: ## only 1 counteractual could be found
#             return 0
#         else:
#             return sum([self._compute_distance(pair[0].cf['X'], pair[1].cf['X'], norm) for pair in comb])/len(comb)

#     def _compute_set_distance(self, cfx1, cfx2, norm):
#         # from scipy.spatial import distance
#         d1 = []
#         for el1 in cfx1:
#             d1.append(min([self._compute_distance(el1.cf['X'], el2.cf['X'], norm) for el2 in cfx2]))

#         d2 = []    
#         for el2 in cfx2:
#             d2.append(min([self._compute_distance(el2.cf['X'], el1.cf['X'], norm) for el1 in cfx1]))

#         return sum(d1)/(2*len(cfx1)) + sum(d2)/(2*len(cfx2))

#     def _compute_set_distance_max(self, cfx1, cfx2, norm):
#         # from scipy.spatial import distance
#         d1 = []
#         for el1 in cfx1:
#             d1.append(min([self._compute_distance(el1.cf['X'], el2.cf['X'], norm) for el2 in cfx2]))

#         d2 = []
#         for el2 in cfx2:
#             d2.append(min([self._compute_distance(el2.cf['X'], el1.cf['X'], norm) for el1 in cfx1]))


#         return 0.5 * (max(d1) + max(d2))


#     def _evaluate_diverse_explanation(self, x, cfx1, cfx2, norm):
#         # Evaluate differrent metrics

#         valid_org = self._check_validity(x, cfx1)
#         valid_noisy = self._check_validity(x, cfx2)

#         # Diversity and set metrics
#         k_dist = self._compute_k_distance(x,cfx1,norm)
#         k_diversity = self._compute_k_diversity(cfx1,norm)
#         match_dist = self._compute_set_distance(cfx1,cfx2,norm)
#         match_b_dist = self._compute_set_distance_max(cfx1,cfx2,norm)

#         return valid_org, valid_noisy, k_dist, k_diversity, match_dist, match_b_dist

#     # def _evaluate_explanation(self, x, cfx1, cfx2,norm):
#     #     # Evaluate differrent metrics


#     #     valid_org = self._check_validity(x, cfx1)
#     #     valid_noisy = self._check_validity(x, cfx2)
#     #     # Metrics for single CFXs
#     #     dist = self._compute_distance(x,cfx1[0].cf['X'],norm)  
#     #     lof = self._compute_lof(cfx1[0].cf['X'])
#     #     dist_cfxs = self._compute_distance(cfx1[0].cf['X'],cfx2[0].cf['X'],norm)

#     #     return valid_org, valid_noisy, dist, lof, dist_cfxs

#     def set_key(self, key):
#         self.current_key = key

#     def evaluate_explanations(self, x, x_noisy, cfx, cfx_noisy, norm):


#         if len(cfx) == 0 or len(cfx_noisy) == 0 is None:
#             self.results[self.current_key] = None
#         else:

#             # Evaluate explanation for factual input
#             valid_org, valid_noisy, k_dist, k_diversity, match_dist, match_b_dist = self._evaluate_diverse_explanation(x, cfx, cfx_noisy,norm)
            
#             if valid_org and valid_noisy and k_diversity is not None:
                
#                 self.results[self.current_key] = {'k_dist': k_dist, 'k_diversity': k_diversity, 'match_dist': match_dist, 'match_b_dist': match_b_dist, "time": self.end_time - self.start_time}
            
#             else:
                
#                 self.results[self.current_key] = None

#             if debug:
#                 print(f"k-distance: {k_dist}")
#                 print(f"k-diversity: {k_diversity}")
#                 print(f"matching distance: {match_dist}")
#                 print(f"matching b-distance: {match_b_dist}")
#                 print(f"Total time: {self.end_time - self.start_time}")
                    


#     # def evaluate_explanation(self, x, x_noisy, cfx, cfx_noisy, norm):


#     #     if len(cfx) == 0 or len(cfx_noisy) == 0 is None:
#     #         self.results[self.current_key] = None
#     #     else:

#     #         # Evaluate explanation for factual input
#     #         valid_org, valid_noisy, dist, lof, dist_cfxs = self._evaluate_explanation(x, cfx, cfx_noisy,norm)

#     #         print(valid_org)
#     #         print(valid_noisy)
            
#     #         if valid_org and valid_noisy:
#     #             self.results[self.current_key] = {'dist': dist, 'lof': lof, 'dist_cfxs': dist_cfxs, "time": self.end_time - self.start_time}
            
#     #         else:
#     #             self.results[self.current_key] = None

#     #         print(f"distance: {dist}")
#     #         print(f"LOF: {lof}")
#     #         print(f"Distance between counterfactuals: {dist_cfxs}")
#     #         print(f"Total time: {self.end_time - self.start_time}")


#     def record_time(self, flag):
#         if flag:
#             self.start_time = time.time()
#         else:
#             self.end_time = time.time()

    
#     def get_summary_set(self, algo):

#         total_exps = len(self.results.values())

#         k_dist = [v['k_dist'] for v in self.results.values() if v is not None]
#         avg_k_dist = np.average(k_dist)
#         median_k_dist = np.median(k_dist)
#         std_k_dist = np.std(k_dist)

#         k_diversity = [v['k_diversity'] for v in self.results.values() if v is not None]
#         avg_k_div = np.average(k_diversity)
#         median_k_div = np.median(k_diversity)
#         std_k_div = np.std(k_diversity)

#         match_dist = [v['match_dist'] for v in self.results.values() if v is not None]
#         avg_match_dist = np.average(match_dist)
#         median_match_dist = np.median(match_dist)
#         std_match_dist = np.std(match_dist)

#         match_b_dist = [v['match_b_dist'] for v in self.results.values() if v is not None]
#         avg_match_b_dist = np.average(match_b_dist)
#         median_match_b_dist = np.median(match_b_dist)
#         std_match_b_dist = np.std(match_b_dist)

#         time = [v['time'] for v in self.results.values() if v is not None]
#         avg_time = np.average(time)
#         median_time = np.median(time)
#         std_time = np.std(time)

#         return f"Algorithm: {algo}.\nValid cfx: {percentage(len(k_dist),total_exps)}.\nAvg k-distance: {avg_k_dist}, {std_k_dist}.\nAvg k-diversity: {avg_k_div}, {std_k_div}.\nAvg match-distance: {avg_match_dist}, {std_match_dist}.\nAvg match-b-distance: {avg_match_b_dist}, {std_match_b_dist}.\nAvg time: {avg_time}, {std_time}.\n"

    # def get_summary_single(self, algo):

    #     total_exps = len(self.results.values())

    #     dist = [v['dist'] for v in self.results.values() if v is not None]
    #     avg_dist = np.average(dist)
    #     median_dist = np.median(dist)

    #     lof = [v['lof'] for v in self.results.values() if v is not None]
    #     avg_lof = np.average(lof)
    #     median_lof = np.median(lof)

    #     dist_cfxs = [v['dist_cfxs'] for v in self.results.values() if v is not None]
    #     avg_dist_cfxs = np.average(dist_cfxs)
    #     median_dist_cfxs = np.median(dist_cfxs)

    #     time = [v['time'] for v in self.results.values() if v is not None]
    #     avg_time = np.average(time)
    #     median_time = np.median(time)

    #     return f"Algorithm: {algo}. Number of valid cfx: {len(dist)}/{total_exps}. Avg distance: {avg_dist}. Avg lof: {avg_lof}. Avg dist_cfxs: {avg_dist_cfxs}. Avg time: {avg_time}.\n " 

        

                





