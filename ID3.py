from math import log, e
import pandas as pd


# class Node:
#     def __init__(self):  # TODO: change names.
#         self.feature = None  # The feature to split by
#         self.next = None  # Next node
#         self.children = None  # Desicition tree branches.


class ID3Algorithm:
    def __init__(self):
        self.train_samples = pd.read_csv("train.csv")
        self.test_samples = pd.read_csv("test.csv")
        self.classifier = None

        self.features_names = self.train_samples.keys()[1:]

    # send the number of positive and negative
    @staticmethod
    def calculate_entropy(pos, neg):
        if neg + pos == 0:
            return 0
        else:
            entropy = -1 * pos / (neg + pos) * log(pos / (neg + pos), 2)
            entropy += -1 * neg / (neg + pos) * log(neg / (neg + pos), 2)
            return entropy

    @staticmethod
    def max_feature(features, values):
        positive, negative = values.loc[values['diagnosis'] == 'B'], values.loc[values['diagnosis'] == 'M']
        entropy = ID3Algorithm.calculate_entropy(positive, negative)

        best_feature= None
        best_IG= float('-inf')
        for feature in features:
            sorted_values= sorted(list(values[feature]), key=lambda x: float(x))
            tresholds= [(i + j) / 2 for i, j in zip(sorted_values[:-1], sorted_values[1:])]
            for th in tresholds:
                positive_lower = (positive.loc[positive[feature] < th]).shape[0]
                doesnt_higher = (negative.loc[negative[feature] <0 th]).shape[0]


    # def __init__(self, features, feature_names, labels):
    #     self.features = features
    #     self.feature_names = feature_names
    #     self.labels = labels  # M or B
    #     self.labelCategories = list(set(labels))  # unique categories
    #     # number of instances of each category
    #     self.labelCategoriesCount = [list(labels).count(x) for x in self.labelCategories]
    #     self.node = None  # nodes
    #     # calculate the initial entropy of the system
    #     self.entropy = self._get_entropy([x for x in range(len(self.labels))])
    #
    # def calculate_entropy(self, samples_ids):
    #     # sorted labels by instance id
    #     labels = [self.labels[i] for i in samples_ids]
    #     # count number of instances of each category
    #     label_count = [labels.count(x) for x in self.labelCategories]
    #     # calculate the entropy for each category and sum them
    #     entropy = sum([-count / len(samples_ids) * math.log(count / len(samples_ids), 2)
    #                    if count else 0
    #                    for count in label_count
    #                    ])
    #
    #     return entropy
    #
    # def calculate_information_gain(self, samples_ids, feature_id):
    #     # calculate total entropy
    #     info_gain = self.calculate_entropy(samples_ids)
    #
    #     # store in a list all the values of the chosen feature
    #     x_features = [self.features[x][feature_id] for x in samples_ids]
    #     labels = [self.labels[i] for i in samples_ids]
    #     features_sorted = sorted(x_features, key=lambda x: x)
    #     tresholds = [(i + j) / 2 for i, j in zip(features_sorted[:-1], features_sorted[1:])]
    #
    #     count_M= labels.count('M')
    #     count_B=labels.count('B')
    #     max_information= float('inf')
    #
    #     for treshold in tresholds:

    # feature_vals = list(set(x_features))
    #
    # # get frequency of each value
    # feature_v_count = [x_features.count(x) for x in feature_vals]
    #
    # # get the feature values ids
    # feature_v_id = [
    #     [x_ids[i]
    #      for i, x in enumerate(x_features)
    #      if x == y]
    #     for y in feature_vals
    # ]
    #
    # # compute the information gain with the chosen feature
    # info_gain_feature = sum([v_counts / len(x_ids) * self.calculate_entropy(v_ids)
    #                          for v_counts, v_ids in zip(feature_v_count, feature_v_id)])
    #
    # info_gain = info_gain - info_gain_feature
    #
    # return info_gain


if __name__ == '__main__':
    id3 = ID3Algorithm()
    id3.buildTree()
    print(str(id3.fit() * 100))
