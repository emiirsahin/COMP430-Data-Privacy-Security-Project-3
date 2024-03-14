import sys
import random

import numpy as np
import pandas as pd
import copy


from collections import Counter
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


###############################################################################
############################# Label Flipping ##################################
###############################################################################
def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    # TODO: You need to implement this function!
    # Implementation of label flipping attack

    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)

    trainValCount = y_train.size    # Total number of training labels
    flipCount = p * trainValCount   # Total number of bits to be flipped
    flipCount = round(flipCount)    # Rounded to the nearest integer
    totalAccuracy = 0   # The total accuracy of the model, divided by the number of iterations at the end

    iterations = 0  # Number of iterations starting at 0

    while(iterations < 100): 

        flipIndices = np.random.choice(trainValCount, flipCount, replace=False) # Generate indices to be flipped
        attackedLabels = y_train.copy() 
        attackedLabels[flipIndices] = 1 - attackedLabels[flipIndices]   # Commence the attack

        if(model_type == "DT"):
            myDEC.fit(X_train, attackedLabels)
            DEC_predict = myDEC.predict(X_test)
            totalAccuracy += accuracy_score(y_test, DEC_predict)

        if(model_type == "LR"):
            myLR.fit(X_train, attackedLabels)
            LR_predict = myLR.predict(X_test)
            totalAccuracy += accuracy_score(y_test, LR_predict)

        if(model_type == "SVC"):
            mySVC.fit(X_train, attackedLabels)
            SVC_predict = mySVC.predict(X_test)
            totalAccuracy += accuracy_score(y_test, SVC_predict)

        iterations += 1
    
    return totalAccuracy/iterations # Return the average accuracy


###############################################################################
########################### Label Flipping Defense ############################
###############################################################################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    # TODO: You need to implement this function!
    # Perform the attack, then the defense, then print the outcome

    trainValCount = y_train.size    # Total number of training labels
    flipCount = p * trainValCount   # Total number of bits to be flipped
    flipCount = round(flipCount)    # Rounded to the nearest integer
    flipIndices = np.random.choice(trainValCount, flipCount, replace=False) # Generate indices to be flipped
    attackedLabels = y_train.copy()

    attackedLabels[flipIndices] = 1 - attackedLabels[flipIndices]   # Commence the attack, no iterations this time

    kmeans = KMeans(n_clusters=2, random_state=0, n_init = 10)   # Initialize the KMeans algorithm
    kmeans.fit(X_train) # Train on unsupervised dataset
    labels = kmeans.predict(X_train)    # Generate labels

    a = 0

    for i,j,k in zip(y_train, labels, attackedLabels):  #
        if i != k:                                      #   Labels that have been flipped in the attack and were correctly identified by the KMeans algorithm
            if j == i:                                  #
                a+=1                                    #

    print(f"Out of {flipCount} flipped data points, {a} were correctly identified.")


###############################################################################
############################# Evasion Attack ##################################
###############################################################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """

    deviationStep = 0.01
    model_name = type(trained_model).__name__
    mode = 0
    if(model_name == "SVC"):
        mode = 1
    if(model_name == "DecisionTreeClassifier"):
        mode = 2

    confidence = np.zeros(2)
    sign = -1 
        
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    if mode == 1:
        confidence = trained_model.decision_function([actual_example])
        if(confidence >= 0):
            sign = 1
        confidence = abs(confidence)
    else:
        confidence = trained_model.predict_proba([actual_example])[0]
        sign = actual_class

        
    if mode == 1:
        loopBool = 1
        while(loopBool):
            coefficient = random.randint(1, 3)/0.5
            modified_example1 = copy.deepcopy(modified_example)
            modified_example2 = copy.deepcopy(modified_example)
            modified_example3 = copy.deepcopy(modified_example)
            modified_example4 = copy.deepcopy(modified_example)
            modified_example5 = copy.deepcopy(modified_example)
            modified_example6 = copy.deepcopy(modified_example)
            modified_example7 = copy.deepcopy(modified_example)
            modified_example8 = copy.deepcopy(modified_example)

            modified_example1[0] = modified_example1[0] + deviationStep * coefficient
            modified_example2[1] = modified_example2[1] + deviationStep * coefficient
            modified_example3[2] = modified_example3[2] + deviationStep * coefficient
            modified_example4[3] = modified_example4[3] + deviationStep * coefficient

            modified_example5[0] = modified_example5[0] - deviationStep * coefficient
            modified_example6[1] = modified_example6[1] - deviationStep * coefficient
            modified_example7[2] = modified_example7[2] - deviationStep * coefficient
            modified_example8[3] = modified_example8[3] - deviationStep * coefficient

            list = [modified_example1, modified_example2, modified_example3,
                    modified_example4, modified_example5, modified_example6,
                    modified_example7, modified_example8]
            
            b = [trained_model.decision_function([modified_example1]), trained_model.decision_function([modified_example2]), trained_model.decision_function([modified_example3])
                , trained_model.decision_function([modified_example4]), trained_model.decision_function([modified_example5]), trained_model.decision_function([modified_example6]),
                trained_model.decision_function([modified_example7]), trained_model.decision_function([modified_example8])]

            for count, i in enumerate(b):
                if(np.sign(i) != sign and np.sign(i) != 0):
                    modified_example = copy.deepcopy(list[count])
                    loopBool = 0
            
            if(loopBool == 0):
                break
             
            absolute_b = [abs(num) for num in b]
            
            idx = absolute_b.index(min(absolute_b))
            
            modified_example = copy.deepcopy(list[idx])
            confidence = abs(trained_model.decision_function([modified_example]))            

        return modified_example
    elif mode == 2:
        loopBool = 1
        
        featureCoefficients = trained_model.feature_importances_
        featureIndices = np.argsort(featureCoefficients)[-2:]

        modified_example1 = copy.deepcopy(modified_example)
        modified_example2 = copy.deepcopy(modified_example)
        
        while(loopBool):
            modified_example1 = copy.deepcopy(modified_example1)
            modified_example2 = copy.deepcopy(modified_example2)
     
            modified_example1[featureIndices[1]] = modified_example1[featureIndices[1]] + deviationStep
            modified_example2[featureIndices[1]] = modified_example2[featureIndices[1]] - deviationStep
            
            modified_example1[featureIndices[0]] = modified_example1[featureIndices[0]] + deviationStep
            modified_example2[featureIndices[0]] = modified_example2[featureIndices[0]] - deviationStep
            
            list = [modified_example1, modified_example2]
            
            b = [trained_model.predict_proba([modified_example1])[0], trained_model.predict_proba([modified_example2])[0]]
            for count, i in enumerate(b):
                if(np.sign(round(i[1])) != sign):
                    modified_example = copy.deepcopy(list[count])
                    loopBool = 0
            
            if(loopBool == 0):
                break

            modified_example1 = copy.deepcopy(list[0])
            modified_example2 = copy.deepcopy(list[1])
            
        return modified_example
    else:
        while(confidence[sign] > confidence[1 - sign]):
            coefficient = random.randint(1, 3)/0.5
            modified_example1 = copy.deepcopy(modified_example)
            modified_example2 = copy.deepcopy(modified_example)
            modified_example3 = copy.deepcopy(modified_example)
            modified_example4 = copy.deepcopy(modified_example)
            modified_example5 = copy.deepcopy(modified_example)
            modified_example6 = copy.deepcopy(modified_example)
            modified_example7 = copy.deepcopy(modified_example)
            modified_example8 = copy.deepcopy(modified_example)

            modified_example1[0] = modified_example1[0] + deviationStep * coefficient
            modified_example2[1] = modified_example2[1] + deviationStep * coefficient
            modified_example3[2] = modified_example3[2] + deviationStep * coefficient
            modified_example4[3] = modified_example4[3] + deviationStep * coefficient

            modified_example5[0] = modified_example5[0] - deviationStep * coefficient
            modified_example6[1] = modified_example6[1] - deviationStep * coefficient
            modified_example7[2] = modified_example7[2] - deviationStep * coefficient
            modified_example8[3] = modified_example8[3] - deviationStep * coefficient

            list = [modified_example1, modified_example2, modified_example3,
                    modified_example4, modified_example5, modified_example6,
                    modified_example7, modified_example8]
            
            b = [trained_model.predict_proba([modified_example1])[0], trained_model.predict_proba([modified_example2])[0], trained_model.predict_proba([modified_example3])[0]
                , trained_model.predict_proba([modified_example4])[0], trained_model.predict_proba([modified_example5])[0], trained_model.predict_proba([modified_example6])[0],
                trained_model.predict_proba([modified_example7])[0], trained_model.predict_proba([modified_example8])[0]]

            differences = [abs(x[1] - x[0]) for x in b]

            idx = differences.index(min(differences))
            
            modified_example = copy.deepcopy(list[idx])
            confidence = trained_model.predict_proba([modified_example])[0]
        return modified_example

def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
########################## Transferability ####################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    # TODO: You need to implement this function!
    # Implementation of transferability evaluation

    DTLRVals = 0
    DTSVCVals = 0
    
    LRDTVals = 0
    LRSVCVals = 0
    
    SVCDTVals = 0
    SVCLRVals = 0
    
    for i in actual_examples:

        modified_modelDT = evade_model(DTmodel, i)
        modified_modelLR = evade_model(LRmodel, i)
        modified_modelSVC = evade_model(SVCmodel, i)
        
        LRActual = LRmodel.predict([i])
        SVCActual = SVCmodel.predict([i])
        DTActual = DTmodel.predict([i])
        
        if(LRmodel.predict([modified_modelDT]) == LRActual):
            DTLRVals+=1
        if(SVCmodel.predict([modified_modelDT]) == SVCActual):
            DTSVCVals+=1
        if(DTmodel.predict([modified_modelLR]) == DTActual):
            LRDTVals+=1
        if(SVCmodel.predict([modified_modelLR]) == SVCActual):
            LRSVCVals+=1
        if(DTmodel.predict([modified_modelSVC]) == DTActual):
            SVCDTVals+=1
        if(LRmodel.predict([modified_modelSVC]) == LRActual):
            SVCLRVals+=1
            
    print("Out of 40 adversarial examples crafted to evade DT :")
    print(f"-> {DTLRVals} of them transfer to LR.")
    print(f"-> {DTSVCVals} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade LR :")
    print(f"-> {LRDTVals} of them transfer to DT.")
    print(f"-> {LRSVCVals} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade SVC :")
    print(f"-> {SVCDTVals} of them transfer to DT.")
    print(f"-> {SVCLRVals} of them transfer to LR.")



###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "PART-1/BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    print("#"*50)
    print("Label flipping attack executions:")
    model_types = ["DT", "LR", "SVC"]
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for p in p_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p)
            print("Accuracy of poisoned", model_type, str(p), ":", acc)

    # Label flipping defense executions:
    print("#" * 50)
    print("Label flipping defense executions:")
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for p in p_vals:
        print("Results with p=", str(p), ":")
        label_flipping_defense(X_train, y_train, p)

    # Evasion attack executions:
    print("#"*50)
    print("Evasion attack executions:")
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    print("#"*50)
    print("Transferability of evasion attacks:")
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])



if __name__ == "__main__":
    main()


