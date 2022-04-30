#Team_JMG
#Binary tree classifier

#imports
import pandas


#filenames for the testing and training data
TEST_FILENAME = "test.csv"
TRAIN_FILENAME = "train.csv"

#the set of features that have a logical order
ordered_features = set()


#A node in a binary tree
class node:
    #the total number of nodes in the tree, used for debugging
    MAX_DEPTH = 7 #having a MAX_DEPTH increases accuracy and reduces runtime
    num_nodes = 0 #the total number of nodes created
    used_set = set() #The set of every feature the tree uses for a split

    #make a node terminal and decide whether to classify as Yes or No
    def make_leaf(self, node_instances):
        self.is_leaf = True
        target_count = node_instances[target].value_counts()
        if target_count["Yes"] > target_count["No"]:
            self.predict_as = "Yes"
        else:
            self.predict_as = "No"
        return

    #the constructor, creates the root node and recursively creates left and right subtrees 
    #node_instances: The collection of instances that made it this far in the tree
    #remaining_features: features that haven't been used yet on the path from the root to this node
    #depth - depth of the node in the tree
    def __init__(self, node_instances, remaining_features, depth):

        
        node.num_nodes += 1 #increment the total number of nodes
        print("Nodes created: ", node.num_nodes)
        self.is_leaf = False #whether or not the node is a leaf

        uniques = node_instances[target].unique() #number of values
        row_count = node_instances.shape[0]
        
        #if instances are all yes or all no, make the node a leaf
        if len(uniques) != 2:
            self.is_leaf = True
            self.predict_as = uniques[0]            
            return

        #if all features were used as splits, max_depth was reached, or the node has less than 10 
        if len(remaining_features) == 0 or depth == node.MAX_DEPTH or node_instances.shape[0] < 10:
            self.make_leaf(node_instances)
            return

        
        self.split_feature = 0
        self.split_feature = ""
        self.left_split = []
        self.right_split = []
        best_gini_gain = 0
        #find the split with the most gini gain
        for feature in remaining_features:
            #choose the split for this feature
            split_left, split_right = choose_split(node_instances, feature)

            #if one of the splits is empty, don't consider this feature
            if len(split_left) == 0 or len(split_right) == 0:
                continue
            
            #find the gini gain for this split
            gini_gain = gini_split(node_instances, feature, split_left, split_right)

            if gini_gain > best_gini_gain:
                self.split_feature = feature
                self.left_split = split_left.copy()
                self.right_split = split_right.copy()
                best_gini_gain = gini_gain

        #if no split_feature was found
        if self.split_feature == "":
            self.make_leaf(node_instances)
            return
        
        
        #copy the insance
        left_df = node_instances.copy(deep = True)
        right_df = node_instances.copy(deep = True)

        #get rid of instances containing values that don't belong to the split
        left_df = left_df[left_df[self.split_feature].isin(self.right_split) == False]
        right_df = right_df[right_df[self.split_feature].isin(self.left_split) == False]

        #if the split will create a node containing no instances
        if(right_df.shape[0] == 0 or left_df.shape[0] == 0):      
            self.make_leaf(node_instances)
            return
        
        
        node.used_set.add(self.split_feature)

        #remove the feature that was used to create the split from remaining_Features
        new_remaining_features = remaining_features.copy()
        new_remaining_features = new_remaining_features.drop(self.split_feature)

        #recursively create left and right subtrees
        self.leftChild = node(left_df, new_remaining_features, depth + 1)
        self.rightChild = node(right_df, new_remaining_features, depth + 1)
        

#classify an instance, tree_node should be the root of a decision tree
def classify(instance, tree_node):
    #once a leaf is reached, predict using the the leaf's predict_as value
    if tree_node.is_leaf:
        return tree_node.predict_as
    #decide which path to take
    if instance[tree_node.split_feature] in tree_node.left_split:
        return classify(instance, tree_node.leftChild)
    else:
        return classify(instance, tree_node.rightChild)
    

#discretizes all numerical features
#requires data to be normalized into the range of 0.0 to 1.0
def discretize(instances):
    #the small negative value is used since the left bound of the bin is exclusive
    #and 0.0 needs to be included
    bins = [-0.0000000001]
    #the data is split into 10 intervals
    NUM_INTERVALS = 10
    for i in range(NUM_INTERVALS):
        bins.append((i+1)*(1 / NUM_INTERVALS))
        
    for i in features:
        #if the feature is numeric
        if not type(instances[i][1]) == type(""):
            ordered_features.add(i)
            #discretize the feature
            instances[i] = pandas.cut(instances[i], bins)
    return instances

#calculates probability of yes given attribute

#finds the probability target = yes given that a categorical feature = value
def prob_yes(instances, feature, value):
    if value not in instances[feature].values:
        return 0
    
    merged = pandas.crosstab(instances[feature],instances[target])
    num_no = merged["No"][value]
    num_yes = merged["Yes"][value]
    return num_yes / (num_yes + num_no)

#finds the average weighted gini of a set of values
#called by gini_split(
def gini(instances, feature, value_set):
    crosstab = pandas.crosstab(instances[feature],instances[target])
    num_no = 0
    num_yes = 0

    #find the number of yes's and no's that appear
    #in instances with value
    for value in value_set:
        if value not in instances[feature].values:
            continue
        num_no += crosstab["No"][value]
        num_yes += crosstab["Yes"][value]
    total = num_no + num_yes
    if total == 0:
        return 0
    weight = total / instances.shape[0]
    prob_yes = num_yes / (total)
    prob_no = 1 - prob_yes
    gini = weight * (1 - (prob_no ** 2 + prob_yes ** 2))
    return gini



#finds the gini gain of a binary split
def gini_split(instances, feature, split_left, split_right):

    left_gini = gini(instances, feature, split_left)
    right_gini = gini(instances, feature, split_right)
    
    gini_impurity = left_gini + right_gini
    
    target_count = instances[target].value_counts()
    row_count = instances.shape[0]
    target_gini = 1 - (target_count["Yes"]/row_count)**2 - (target_count["No"]/row_count)**2
    gini_gain = target_gini - gini_impurity
    
    return gini_gain
    
#choose how to split a feature 
def choose_split(instances, feature):
    #t
    split_left = []
    split_right = []

    #for unordered features
    if feature not in ordered_features:
        #for each value the instance can have
        for value in instances[feature].unique():
            if prob_yes(instances, feature, value) > 0.5:
                split_left.append(value)
            else:
                split_right.append(value)
    #for ordered features
    else:
        best_gain = 0
        #try every possible split
        #choose the one with the most gini gain
        for i in range(len(possible_split_left)):
            temp_left = possible_split_left[i]
            temp_right = possible_split_right[i]
                    
            gini_gain = gini_split(instances, feature, temp_left, temp_right)
            if gini_gain > best_gain:
                best_gain = gini_gain
                split_left = temp_left.copy()
                split_right = temp_right.copy()
    return [split_left, split_right]





#get data into instances
print("Opening " + TRAIN_FILENAME)
data = pandas.read_csv(TRAIN_FILENAME)


target = data.columns[-1] #target variable
features = data.columns[0:-1] #features variables



#discretize the data
data = discretize(data)


#"Sunshine" is used as an arbitrary numerical feature
data_intervals = data["Sunshine"].unique().categories


#find all of the possible ordered splits for numerical features
#since the data is normalized, all data is discretized into the same ranges
possible_split_left = []
possible_split_right = []
for i in range(len(data_intervals)):
    temp_left = []
    temp_right = []
    for j in range(len(data_intervals)):
        if j <= i:
            temp_left.append(data_intervals[j])
        else:
            temp_right.append(data_intervals[j])
    possible_split_left.append(temp_left.copy())
    possible_split_right.append(temp_right.copy())



#create the tree
print("\nCreating Tree. This will take around 10 mintues to create ~250 nodes.")
root = node(data, features, 0)





#testing data

print("\nOpening " + TEST_FILENAME)
test_data = pandas.read_csv(TEST_FILENAME)

test_data = discretize(test_data)

tp = 0
fp = 0
tn = 0
fn = 0

print("Testing model")

#for each instance of test data
for i in range(test_data.shape[0]):
    row = test_data.iloc[i] #get the ith instance
    actual = row[target]
    predicted = classify(row, root)
    #update confusion matrix values
    if actual == "Yes":
        if predicted == "Yes":
            tp += 1
        else:
            fn += 1
    if actual == "No":
        if predicted == "Yes":
            fp += 1
        else:
            tn += 1

#output confusion matrix related information
print("True Positive: ", tp)
print("True Negative: ", tn)
print("False Positive: ", fp)
print("False Negative: ", fn)

print("\nAccuracy: ", (tp + tn)/(tp + tn + fp + fn))
print("Senstivity: ", tp/(tp+fn))
print("Specificity: ", tn/(tn+fp))
print("Precision: ", tp/(tp+fp))

print("\n end of program")

