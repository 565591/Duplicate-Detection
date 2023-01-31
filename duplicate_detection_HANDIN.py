# First we import the data
import json
import math
import re
import sympy
import numpy as np
import kshingle as ks
import random
from strsimpy.qgram import QGram
from math import comb
from sklearn.cluster import AgglomerativeClustering
qgram = QGram(3)
from strsimpy.levenshtein import Levenshtein
levenshtein = Levenshtein()
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
normalized_levenshtein = NormalizedLevenshtein()
from matplotlib import pyplot as plt


# We create a function which allows us to create different hash functions
# We create a prime number bigger than the size of the signature matrix. This makes sure we do not get collisions when simulating permutations
# x is the value that you want to hash
# s is the size of the signature matrix
# b & a allow us to create different hash functions
def hashfunction(x, s, b, a):
    prime = sympy.nextprime(s + 500)
    x = int(x)
    b = int(b)
    number = (b * x + a) % prime
    return number


# I CREATE SOME FUNCTIONS THAT WILL BE USED BY THE "performOneBootstrap" FUNCTION THAT I WILL DEFINE LATER:
# I create a function which returns the 3-gram similarity of two strings:
def stringSimilarity(stringA, stringB):
    shinglesA = ks.shingleset_list(stringA, [3])
    shinglesB = ks.shingleset_list(stringB, [3])
    numerator = len(shinglesA) + len(shinglesB) - qgram.distance(stringA, stringB)
    denominator = len(shinglesA) + len(shinglesB)
    toReturn = 0
    # NOTE YOU RETURN ZERO WHEN THERE ARE NO 3GRAM ELEMENTS
    if denominator > 0:
        toReturn = numerator / denominator
    return toReturn


# I create a function which extracts model words from a string
def extractModelWordsFromString(string):
    listmod = string.split()
    modelWordsSet2 = set()
    for words in listmod:
        checker = re.match("[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*", words)
        if checker:
            modelWordsSet2.add(words)
    return modelWordsSet2


# I create a function which takes a dissimilarity matrix and analyzes the clusters
def performClustering(matrix, matrixToProductMapper, threshold):
    dictionaryWithClusters = dict()
    model = AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=threshold,
                                    n_clusters=None)
    model.fit(matrix)

    for i in range(len(np.unique(model.labels_))):
        dictionaryWithClusters[i] = list()
        for index, element in enumerate(model.labels_):
            if element == i:
                dictionaryWithClusters[i].append(matrixToProductMapper[index])

    for key in dictionaryWithClusters:
        dictionaryWithClusters[key].sort()

    return dictionaryWithClusters


# We prepare some auxiliary function for titleModelWordsDistance
def calcCosineSim(string1, string2):
    splittedString1 = string1.split()
    splittedString2 = string2.split()
    splittedStringSet1 = set(splittedString1)
    splittedStringSet2 = set(splittedString2)

    cosineSim = len(splittedStringSet1.intersection(splittedStringSet2)) / (
            math.sqrt(len(splittedStringSet1)) * math.sqrt(len(splittedStringSet2)))
    return cosineSim


# A function which calculates the average normalized Levenshtein similarity between two sets of model words
def avgLvSim(setA, setB):
    listA = list(setA)
    listB = list(setB)
    averageLvSim = 0
    scalingDenominator = 0
    numerator = 0

    for elementA in listA:
        for elementB in listB:
            scalingDenominator = scalingDenominator + len(elementA) + len(elementB)
            numerator = numerator + (1 - normalized_levenshtein.distance(elementA, elementB))*(len(elementA) + len(elementB))



    averageLvSim = numerator / scalingDenominator

    return averageLvSim


# Similar to avgLvSm but here we only include pairs of products where the numerical part is the same and the
# non-numerical part is approximately the same.
# The function returns -1 if there is no such match within all pairs of modelwords



def avgLvSimMW(modelWordsList1, modelWordsList2, parameterStringSimilarity):
    scalingDenominator = 0
    mathingModelswords = 0
    numerator = 0
    for index1, element1 in enumerate(modelWordsList1):
        for element2 in modelWordsList2:
            element1_nonNumerical = re.findall(r'(\D+)', element1)
            element2_nonNumerical = re.findall(r'(\D+)', element2)
            element1_numerical = re.findall(r'(\d+)', element1)
            element2_numerical = re.findall(r'(\d+)', element2)

            levenshteinDistance_nonnumerical = normalized_levenshtein.distance(element1_nonNumerical,
                                                                               element2_nonNumerical)

            if levenshteinDistance_nonnumerical < parameterStringSimilarity and element1_numerical == element2_numerical:
                mathingModelswords = mathingModelswords + 1
                scalingDenominator = scalingDenominator + len(element1) + len(element2)
                numerator = numerator + (1 - normalized_levenshtein.distance(element1, element2))*(len(element1) + len(element2))

    if mathingModelswords == 0:
        return -1
    else:
        return numerator/scalingDenominator





def titleModelWordsDistance(title_1, title_2, alpha, beta, delta, parameterStringSimilarity):
    # First we clean the product titles making them more easy to compare
    title_1 = title_1.replace('and', ' ')
    title_1 = title_1.replace('or', ' ')
    title_1 = title_1.replace('&', ' ')
    title_1 = title_1.replace('/', ' ')
    title_1 = title_1.replace('-', ' ')
    title_2 = title_2.replace('and', ' ')
    title_2 = title_2.replace('or', ' ')
    title_2 = title_2.replace('&', ' ')
    title_2 = title_2.replace('/', ' ')
    title_2 = title_2.replace('-', ' ')

    nameCosineSim = calcCosineSim(title_1, title_2)
    if nameCosineSim > alpha:
        return 1
    modelWordsSet1 = extractModelWordsFromString(title_1)
    modelWordsSet2 = extractModelWordsFromString(title_2)
    modelWordsList1 = list(modelWordsSet1)
    modelWordsList2 = list(modelWordsSet2)

    # For each combination of model words we extract the numerical part and the non-numerical part for both model words
    # We look whether there is combination where the non-numerical parts are approximately the same and the numerical part is different
    # This is an indication that we are not dealing with the same product
    badMatches = 0
    for index1, element1 in enumerate(modelWordsList1):
        for index2, element2 in enumerate(modelWordsList2):
            element1_nonNumerical = re.findall(r'(\D+)',element1)
            element2_nonNumerical = re.findall(r'(\D+)',element2)
            element1_numerical = re.findall(r'(\d+)',element1)
            element2_numerical = re.findall(r'(\d+)',element2)
            levenshteinDistance_nonnumerical = normalized_levenshtein.distance(element2_nonNumerical,
                                                                                   element1_nonNumerical)

            if levenshteinDistance_nonnumerical < parameterStringSimilarity and element1_numerical != element2_numerical:
                badMatches = badMatches + 1
    if badMatches > 0:
        return -1
    else:
        # We calculate initial product name similarity (which might get changed later)
        finalNameSim = beta * nameCosineSim + (1 - beta) * avgLvSim(modelWordsSet1, modelWordsSet2)

        # We check whether there is a combination of models words in the two sets where the non-numerical part matches
        # approximately and the numerical parts exactly. If this is the case we update our final name similarity
        # by giving more weight to the model words combinations that match
        # See function avgLvSimMW for more details
        modelWordSimVal = avgLvSimMW(modelWordsList1, modelWordsList2,parameterStringSimilarity)
        if modelWordSimVal == -1:
            return finalNameSim
        else:
            finalNameSim = delta * modelWordSimVal + (1 - delta) * finalNameSim
            return finalNameSim


def getPredictedPairsFromClustering(dictionaryWithClusters):
    predictedPairs = list()
    for key in dictionaryWithClusters:
        if len(dictionaryWithClusters[key]) > 1:
            for indexA, element in enumerate(dictionaryWithClusters[key]):
                for indexB in range(indexA + 1, len(dictionaryWithClusters[key])):
                    predictedPairs.append((element, dictionaryWithClusters[key][indexB]))

    return predictedPairs


# I CREATE A FUNCTION WHICH TAKES AS INPUT THE THRESHOLD VALUE T
# THE FUNCTION CALCULATES THE NUMBER OF BANDS AND ROWS ASSOCIATED WITH THE THRESHOLD VALUE T
# WITHIN THIS FUNCTION 5 BOOTSTRAPS ARE TAKEN
# FOR EACH OF THESE BOOTSTRAPS:
# -> LOCALITY SENSITIVE HASHING IS APPLIED ON EACH OF THESE BOOTSTRAPS TO FIND CANDIDATE PAIRS
# -> THE FRACTION OF COMPARISONS MADE IS CALCULATED
# -> THE F1* MEASURE IS CALCULATED
# -> THE DISTANCE MATRIX IS CALCULATED THROUGH MSM ALGORITHM
# -> HIERARCHICAL CLUSTERING IS DONE
# -> THE F1 MEASURE IS CALCULATED





def findOptimalHyperParameters_ONLYEPSILON(t,data, bootStrapTrainingList):
    # mu is the parameter that determines the weight of titlesim in MSM
    mu = 0.65
    # gamma is the parameter in MSM that determines string similarity
    gamma = 0.75

    # parameters for titleModelWordsDistance:
    alpha = 0.602
    beta = 0
    delta = 0.4

    # parameter used in both titleModelWordsDistance & avgLvSimMW
    parameterStringSimilarity = 0.3

    # parameter epsilon/threshold for clustering
    threshold = 0.522

    # WE CREATE MAPS WHICH ALLOWS US TO LINK OBSERVATIONS FROM OUR BOOTSTRAPS TO THE ORIGINAL OBSERVATIONS
    # This one will return the index in the bootstrap if you give the original number
    productToMatrixMapper = dict()
    # This one will return the original number if you give the index in the bootstrap
    matrixToProductMapper = dict()

    for index, product in enumerate(bootStrapTrainingList):
        productToMatrixMapper[product] = index
        matrixToProductMapper[index] = product

    # HERE I WILL ADD A NEW CONDENSED FEATURES MAP WHERE I ONLY SELECT THE RELEVANT FEATURES THAT I WILL BE USING IN LSH
    featuresToConsider = ['wifi','dimension','size', 'width', 'height', 'depth', 'ratio', 'weight', 'resolution', 'rate']
    for relevantKeys in bootStrapTrainingList:
        data[relevantKeys]['condensedFeaturesMap'] = dict()
        for valueKey in data[relevantKeys]['featuresMap']:
            boolean_checker = False
            for features in featuresToConsider:
                if features in valueKey.lower():
                    boolean_checker = True
            if boolean_checker:
                data[relevantKeys]['condensedFeaturesMap'][str(valueKey)] = data[relevantKeys]['featuresMap'][valueKey]

    # NEXT WE CREATE UNIFORM BINARY VECTOR REPRESENTATIONS OF OUR PRODUCTS
    # First we extract model words from the title
    # We also extract k-grams from the remainder of the title (where the model words are removed)
    modelWordsSet = set()
    for key in bootStrapTrainingList:
        lista = data[key]['title'].split()
        for words in lista:
            checker = re.match("[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*", words)
            if checker:
                modelWordsSet.add(words)
        for attributes in data[key]['condensedFeaturesMap']:
            valueString = data[key]['condensedFeaturesMap'][attributes]
            valueList = valueString.split()
            for valueWords in valueList:
                valueChecker = re.match("(^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$)", valueWords)
                if valueChecker:
                    valueWordsRemovedNonNumerical = valueWords
                    for c in valueWordsRemovedNonNumerical:
                        if (not c.isdigit()) and c != '.':
                            valueWordsRemovedNonNumerical = valueWordsRemovedNonNumerical.replace(c, '')
                    modelWordsSet.add(valueWordsRemovedNonNumerical)

    # Here i make sure my set will be an ordered list
    modelWordsList = sorted(modelWordsSet)



    # Here we actually create the uniform vector representations
    for key in bootStrapTrainingList:
        productRepresentation = [0] * len(modelWordsList)
        for index, element in enumerate(modelWordsList):
            if element in data[key]['title']:
                productRepresentation[index] = 1

            for attributes in data[key]['condensedFeaturesMap']:
                if element in data[key]['condensedFeaturesMap'][attributes]:
                    productRepresentation[index] = 1

        data[key]['productRepresentation'] = productRepresentation


    # NEXT WE TRANSFORM THE UNIFORM BINARY PRODUCT REPRESENTATIONS INTO SIGNATURE VECTORS
    # These signature vectors will be computed by an approximation of permutations for efficiency

    # first we initialize the amount of signatures and the signature vectors in our dictionary
    s = round(0.5 * len(modelWordsList))
    for key in data:
        data[key]['signature'] = [math.inf] * s

    # Now we generate the signatures
    # POSSIBLY MISTAKES HERE? HARD TO TEST. IF EVERYTHING DOESNT WORK THE MISTAKE IS PROBABLY HERE (but looks good..)
    InputHashFunction_a = [0] * s
    InputHashFunction_b = [0] * s
    for i in range(s):
        randomb = random.randrange(1, 100000000)
        randoma = random.randrange(1, 100000000)
        InputHashFunction_a[i] = randoma
        InputHashFunction_b[i] = randomb

    for element in range(1, len(modelWordsList) + 1):
        outputHashFunction = [0] * s
        for index, i in enumerate(InputHashFunction_a):
            outputHashFunction[index] = hashfunction(element, len(modelWordsList), InputHashFunction_b[index], i)
        for key in bootStrapTrainingList:
            if data[key]['productRepresentation'][element - 1] == 1:
                for j, value in enumerate(outputHashFunction):
                    if value < data[key]['signature'][j]:
                        data[key]['signature'][j] = value

    # NOW WE BUILD THE LOCALITY SENSITIVE HASHING ALGORITHM
    # The threshold t defines how similar documents have to be in order to be regarded as a similar pair
    # With t we can determine the number of bands (b) and the number of rows per band (r) needed
    placeholder = math.inf
    best = 0
    for r in range(1, s):
        computation = abs((r / s) ** (1 / r) - t)
        if computation < placeholder:
            placeholder = computation
            best = r

    r = best
    b = int(s / r)

    # Now we add as an additional feature in our dictionary which includes the bands of the signature vector
    for key in bootStrapTrainingList:
        data[key]['chunkedsignature'] = np.array_split(data[key]['signature'], b)

    # We turn the numbers in a band into a string which will help with our implementation of locality sensitive hashing
    for key in bootStrapTrainingList:
        data[key]['chunkedsignature_asStrings'] = list()
        for band in range(b):
            intermediateList = [str(i) for i in data[key]['chunkedsignature'][band]]
            string_band = int(''.join(intermediateList))
            data[key]['chunkedsignature_asStrings'].append(string_band)

    # NOW WE PERFORM LSH AND RETRIEVE THE CANDIDATE PAIRS
    # now we will put the chunks into buckets.
    # Note the number of buckets are chosen larger than the number of products in order to avoid collisions if the bands are not exactly the same
    candidateset = set()
    for band in range(b):
        bandDictionary = dict()
        for key in bootStrapTrainingList:
            if data[key]['chunkedsignature_asStrings'][band] in bandDictionary.keys():
                bandDictionary[data[key]['chunkedsignature_asStrings'][band]].append(key)

            else:
                bandDictionary[data[key]['chunkedsignature_asStrings'][band]] = list()
                bandDictionary[data[key]['chunkedsignature_asStrings'][band]].append(key)

        for bucketKeys in bandDictionary:
            for indexA, a in enumerate(bandDictionary[bucketKeys]):
                for indexB in range(indexA + 1, len(bandDictionary[bucketKeys])):
                    candidateset.add((a, bandDictionary[bucketKeys][indexB]))

    # Now we turn our candidate set to a list
    candidateList = list(candidateset)
    for index, tup in enumerate(candidateList):
        candidateList[index] = tuple(sorted(tup))
    candidateset = set(candidateList)
    candidateList = list(candidateset)

    best_threshold = 0
    MAXIMUM_F1 = 0
    thresholds = [0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.522,0.53,0.54,0.55]
    numberOfCombinations = len(thresholds)
    print('NUMBER OF HYPERPARAMETERS COMBINATIONS EVALUATED:')
    print(numberOfCombinations)
    counter = 0
    for threshold_iterate in thresholds:
        counter = counter + 1
        print(counter)
        # mu is the parameter that determines the weight of titlesim in MSM
        mu_iterate = 0.65
        beta_iterate = 0
        delta_iterate = 0.4
        alpha_iterate = 0.602
        gamma_iterate = 0.756
        parameterStringSimilarity_iterate = 0.3
        # NOW WE CREATE THE DISSIMILARITY MATRIX
        dissimilarityMatrix = np.asarray(
            np.ones((len(bootStrapTrainingList), len(bootStrapTrainingList))) * 1000000)

        for candidates in candidateList:
            if ((data[candidates[0]]['featuresMap']['Brand'] !=
                 data[candidates[1]]['featuresMap']['Brand']) and
                    data[candidates[0]]['featuresMap']['Brand'] != 'nobrand' and
                    data[candidates[1]]['featuresMap'][
                        'Brand'] != 'nobrand'):

                dissimilarityMatrix[productToMatrixMapper[candidates[0]], productToMatrixMapper[
                    candidates[1]]] = 1000000

            elif data[candidates[0]]['shop'] == data[candidates[1]]['shop']:
                dissimilarityMatrix[productToMatrixMapper[candidates[0]], productToMatrixMapper[
                    candidates[1]]] = 1000000

            else:
                sim = 0
                avgSim = 0
                m = 0
                w = 0
                nonMatchingKeysA = set(data[candidates[0]]['featuresMap'].keys())
                nonMatchingKeysB = set(data[candidates[1]]['featuresMap'].keys())
                for keysA in data[candidates[0]]['featuresMap']:
                    for keysB in data[candidates[1]]['featuresMap']:
                        keySim = stringSimilarity(keysA, keysB)
                        if keySim > gamma_iterate:
                            valueSim = stringSimilarity(
                                data[candidates[0]]['featuresMap'][keysA],
                                data[candidates[1]]['featuresMap'][keysB])
                            weight = keySim
                            sim = sim + weight * valueSim
                            m = m + 1
                            w = w + weight
                            nonMatchingKeysA.remove(keysA)
                            nonMatchingKeysB.remove(keysB)

                if w > 0:
                    avgSim = sim / w

                setModelWordsA = set()
                setModelWordsB = set()
                for keyz1 in nonMatchingKeysA:
                    for bla in extractModelWordsFromString(
                            data[candidates[0]]['featuresMap'][keyz1]):
                        setModelWordsA.add(bla)

                for keyz2 in nonMatchingKeysB:
                    for bla2 in extractModelWordsFromString(
                            data[candidates[1]]['featuresMap'][keyz2]):
                        setModelWordsB.add(bla2)

                intersectionBoth = setModelWordsA.intersection(setModelWordsB)
                if len(setModelWordsA) > 0 and len(setModelWordsB) > 0:
                    mwPerc = (len(intersectionBoth) * 2) / (
                            len(setModelWordsA) + len(setModelWordsB))
                else:
                    mwPerc = 0

                titleSim = titleModelWordsDistance(data[candidates[0]]['title'],
                                                   data[candidates[1]]['title'], alpha_iterate,
                                                   beta_iterate, delta_iterate,
                                                   parameterStringSimilarity_iterate)

                if titleSim == -1:
                    theta1 = m / min(len(data[candidates[0]]['featuresMap'].keys()),
                                     len(data[candidates[1]]['featuresMap'].keys()))
                    theta2 = 1 - theta1
                    hSIM = theta1 * avgSim + theta2 * mwPerc
                    dist = 1 - hSIM

                else:
                    theta1 = (1 - mu_iterate) * (m / min(len(data[candidates[0]]['featuresMap'].keys()),
                                                         len(data[candidates[1]][
                                                                 'featuresMap'].keys())))
                    theta2 = 1 - mu_iterate - theta1
                    hSIM = theta1 * avgSim + theta2 * mwPerc + mu_iterate * titleSim
                    dist = 1 - hSIM

                dissimilarityMatrix[productToMatrixMapper[candidates[0]], productToMatrixMapper[
                    candidates[1]]] = dist

        # Now i make sure the other symmetric distances are also filled in.
        for i in range(dissimilarityMatrix.shape[0]):
            for j in range(i + 1, dissimilarityMatrix.shape[0]):
                if not dissimilarityMatrix[i, j] == 1000000:
                    dissimilarityMatrix[j, i] = dissimilarityMatrix[i, j]

        # NOW WE PERFORM THE CLUSTERING FOR ALL DISSIMILARITY MATRICES AND CALCULATE THE F1 SCORE
        dictionaryWithClusters = performClustering(dissimilarityMatrix, matrixToProductMapper,
                                                   threshold_iterate)
        listWithPredictedPairs = getPredictedPairsFromClustering(dictionaryWithClusters)

        TruePositives = 0
        FalsePositives = 0
        FalseNegatives = 0

        for element in listWithPredictedPairs:
            if data[element[0]]['modelID'] == data[element[1]]['modelID']:
                TruePositives = TruePositives + 1
            else:
                FalsePositives = FalsePositives + 1

        allPossiblePairsClust = set()
        for index1, element in enumerate(bootStrapTrainingList):
            for index2 in range(index1 + 1, len(bootStrapTrainingList)):
                allPossiblePairsClust.add((element, bootStrapTrainingList[index2]))

        predictedToBeNegative = allPossiblePairsClust.difference(set(listWithPredictedPairs))
        predictedToBeNegative = list(predictedToBeNegative)

        for pred in predictedToBeNegative:
            if data[pred[0]]['modelID'] == data[pred[1]]['modelID']:
                FalseNegatives = FalseNegatives + 1

        if TruePositives > 0 or FalsePositives > 0:
            precision = TruePositives / (TruePositives + FalsePositives)
        else:
            precision = 0

        if TruePositives > 0 or FalseNegatives > 0:
            recall = TruePositives / (TruePositives + FalseNegatives)
        else:
            recall = 0

        # NOTE SOMETIMES WE GET 0 CANDIDATES!
        if precision > 0 or recall > 0:
            F1 = (2 * precision * recall) / (precision + recall)
        else:
            F1 = 0

        if F1 > MAXIMUM_F1:
            best_mu = mu_iterate
            best_beta = beta_iterate
            best_alpha = alpha_iterate
            best_delta = delta_iterate
            best_gamma = gamma_iterate
            best_parameterStringSimilarity = parameterStringSimilarity_iterate
            best_threshold = threshold_iterate

    return [0.65,best_threshold,0.756,0.4,0.602,0,0.3]



#You can use these parameters instead of tuning (retrieved from papers):
parametersFromPapers = [0.65,0.522,0.756,0.4,0.602,0,0.3]

def performOneBootstrap(t):
    # WE IMPORT DATA AND WE PUT THE DATA IN THE APPROPRIATE FORMAT AND DO PRELIMINARY CLEANING.
    with open("/Users/safuat/Desktop/cs/TVs-all-merged.json") as f:
        data_notnice = json.load(f)

    data = dict()
    # First i put everything in a new dictionary where the keys will be an index for the product
    counter = 0
    for key in data_notnice:
        for index, element in enumerate(data_notnice[key]):
            counter = counter + 1
            data[counter] = data_notnice[key][index]

    # we start off by cleaning the data in the title as described by reference 3
    for key in data:
        data[key]['title'] = data[key]['title'].replace("/", " ")
        data[key]['title'] = data[key]['title'].replace('\\', ' ')
        data[key]['title'] = data[key]['title'].replace("-", " ")
        data[key]['title'] = data[key]['title'].replace('Inch', 'inch')
        data[key]['title'] = data[key]['title'].replace('inches', 'inch')
        data[key]['title'] = data[key]['title'].replace('"', 'inch')
        data[key]['title'] = data[key]['title'].replace('-inch', 'inch')
        data[key]['title'] = data[key]['title'].replace(' inch', 'inch')
        data[key]['title'] = data[key]['title'].replace('class', '')
        data[key]['title'] = data[key]['title'].replace('”', 'inch')
        data[key]['title'] = data[key]['title'].replace('inchdiagonal', 'inch')
        #data[key]['title'] = data[key]['title'].replace('.0', '')
        data[key]['title'] = data[key]['title'].replace("'", 'inch')
        data[key]['title'] = data[key]['title'].replace('Hertz', 'hz')
        data[key]['title'] = data[key]['title'].replace('hertz', 'hz')
        data[key]['title'] = data[key]['title'].replace('Hz', 'hz')
        data[key]['title'] = data[key]['title'].replace('HZ', 'hz')
        data[key]['title'] = data[key]['title'].replace(' hz', 'hz')
        data[key]['title'] = data[key]['title'].replace('-hz', 'hz')
        data[key]['title'] = data[key]['title'].lower()
        # These ones are new. this is important since we otherwise would have model words like (54.64inch THIS WAS NOT IN THE PAPER
        data[key]['title'] = data[key]['title'].replace('(', ' ')
        data[key]['title'] = data[key]['title'].replace(')', ' ')
        data[key]['title'] = data[key]['title'].replace(',', ' ')
        data[key]['title'] = data[key]['title'].replace('Newegg.com', '')
        data[key]['title'] = data[key]['title'].replace('TheNerds.net', '')

    # Now i will make sure all products contain a valid brand in the features map.

    # First we add a key-value pair for the products without a brand in the features map
    for key in data:
        if (not 'Brand' in data[key]['featuresMap']):
            data[key]['featuresMap']['Brand'] = 'nobrand'

    # I convert all brand names no lowercase
    for key in data:
        data[key]['featuresMap']['Brand'] = data[key]['featuresMap']['Brand'].lower()

    # I make sure the problematic brands lg & vca are in the same format
    for key in data:
        if data[key]['featuresMap']['Brand'] == 'lg electronics':
            data[key]['featuresMap']['Brand'] = 'lg'

        if data[key]['featuresMap']['Brand'] == 'jvc tv':
            data[key]['featuresMap']['Brand'] = 'jvc'

    # I create a set with all unique brand contained in the dataset
    setofbrands = set()
    for key in data:
        setofbrands.add(data[key]['featuresMap']['Brand'])
    setofbrands.discard('nobrand')
    # I add popular brands which did not appear in the featuresMap (i did this manually, but can be done from online lists)
    setofbrands.add('tcl')
    setofbrands.add('insignia')
    setofbrands.add('avue')
    setofbrands.add('optoma')
    setofbrands.add('dynex')
    setofbrands.add('mitsubishi')
    setofbrands.add('contex')

    # If a product has no brand, we extract the brand from the title and add it to the featuresMap
    for key in data:
        if (data[key]['featuresMap']['Brand'] == 'nobrand'):
            for brand in setofbrands:
                if (brand in data[key]['title']):
                    data[key]['featuresMap']['Brand'] = brand

    # WE TAKE A BOOTSTRAP FROM THE ORIGINAL DATASET
    bootStrapTrainingSet = set()
    for iteration in data:
        bootStrapTrainingSet.add(random.choice(list(data)))
    allkeys = set(data.keys())
    bootStrapTestList = list(allkeys.difference(bootStrapTrainingSet))
    bootStrapTestList.sort()
    bootStrapTrainingList = list(bootStrapTrainingSet)
    bootStrapTrainingList.sort()

    ######                                                                                                  ########
    ###################### HERE WE WORK WITH THE TRAINING SET TO FIND OPTIMAL HYPERPARAMETERS ######################
    ######                                                                                                  ########

    tunedParameters = findOptimalHyperParameters_ONLYEPSILON(t,data, bootStrapTrainingList)
    #tunedParameters = [0.65, 0.522, 0.756, 0.4, 0.602, 0, 0.3]
    #print(tunedParameters)



    ###############################################################################################################

    # WE CREATE MAPS WHICH ALLOWS US TO LINK OBSERVATIONS FROM OUR BOOTSTRAPS TO THE ORIGINAL OBSERVATIONS
    # This one will return the index in the bootstrap if you give the original number
    productToMatrixMapper = dict()
    # This one will return the original number if you give the index in the bootstrap
    matrixToProductMapper = dict()

    for index, product in enumerate(bootStrapTestList):
        productToMatrixMapper[product] = index
        matrixToProductMapper[index] = product

    # HERE I WILL ADD A NEW CONDENSED FEATURES MAP WHERE I ONLY SELECT THE RELEVANT FEATURES THAT I WILL BE USING IN LSH
    featuresToConsider = ['wifi','dimension','size', 'width', 'height', 'depth', 'ratio', 'weight', 'resolution', 'rate']
    for relevantKeys in bootStrapTestList:
        data[relevantKeys]['condensedFeaturesMap'] = dict()
        for valueKey in data[relevantKeys]['featuresMap']:
            boolean_checker = False
            for features in featuresToConsider:
                if features in valueKey.lower():
                    boolean_checker = True
            if boolean_checker:
                data[relevantKeys]['condensedFeaturesMap'][str(valueKey)] = data[relevantKeys]['featuresMap'][valueKey]




    # NEXT WE CREATE UNIFORM BINARY VECTOR REPRESENTATIONS OF OUR PRODUCTS
    # First we extract model words from the title
    # We also extract k-grams from the remainder of the title (where the model words are removed)
    modelWordsSet = set()
    for key in bootStrapTestList:
        lista = data[key]['title'].split()
        for words in lista:
            checker = re.match("[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*", words)
            if checker:
                modelWordsSet.add(words)
        for attributes in data[key]['condensedFeaturesMap']:
            valueString = data[key]['condensedFeaturesMap'][attributes]
            valueList = valueString.split()
            for valueWords in valueList:
                valueChecker = re.match("(^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$)", valueWords)
                if valueChecker:
                    valueWordsRemovedNonNumerical = valueWords
                    for c in valueWordsRemovedNonNumerical:
                        if (not c.isdigit()) and c != '.':
                            valueWordsRemovedNonNumerical = valueWordsRemovedNonNumerical.replace(c, '')
                    modelWordsSet.add(valueWordsRemovedNonNumerical)


    # Here i make sure my set will be an ordered list
    modelWordsList = sorted(modelWordsSet)



    # Here we actually create the uniform vector representations
    for key in bootStrapTestList:
        productRepresentation = [0] * len(modelWordsList)
        for index, element in enumerate(modelWordsList):
            if element in data[key]['title']:
                productRepresentation[index] = 1

            for attributes in data[key]['condensedFeaturesMap']:
                if element in data[key]['condensedFeaturesMap'][attributes]:
                    productRepresentation[index] = 1


        data[key]['productRepresentation'] = productRepresentation




    # NEXT WE TRANSFORM THE UNIFORM BINARY PRODUCT REPRESENTATIONS INTO SIGNATURE VECTORS
    # These signature vectors will be computed by an approximation of permutations for efficiency

    # first we initialize the amount of signatures and the signature vectors in our dictionary
    s = round(0.5 * len(modelWordsList))
    for key in data:
        data[key]['signature'] = [math.inf] * s

    # Now we generate the signatures
    # POSSIBLY MISTAKES HERE? HARD TO TEST. IF EVERYTHING DOESNT WORK THE MISTAKE IS PROBABLY HERE (but looks good..)
    InputHashFunction_a = [0] * s
    InputHashFunction_b = [0] * s
    for i in range(s):
        randomb = random.randrange(1, 100000000)
        randoma = random.randrange(1, 100000000)
        InputHashFunction_a[i] = randoma
        InputHashFunction_b[i] = randomb

    for element in range(1, len(modelWordsList) + 1):
        outputHashFunction = [0] * s
        for index, i in enumerate(InputHashFunction_a):
            outputHashFunction[index] = hashfunction(element, len(modelWordsList), InputHashFunction_b[index], i)
        for key in bootStrapTestList:
            if data[key]['productRepresentation'][element - 1] == 1:
                for j, value in enumerate(outputHashFunction):
                    if value < data[key]['signature'][j]:
                        data[key]['signature'][j] = value

    # NOW WE BUILD THE LOCALITY SENSITIVE HASHING ALGORITHM
    # The threshold t defines how similar documents have to be in order to be regarded as a similar pair
    # With t we can determine the number of bands (b) and the number of rows per band (r) needed
    placeholder = math.inf
    best = 0
    for r in range(1, s):
        computation = abs((r / s) ** (1 / r) - t)
        if computation < placeholder:
            placeholder = computation
            best = r

    r = best
    b = int(s / r)

    # Now we add as an additional feature in our dictionary which includes the bands of the signature vector
    for key in bootStrapTestList:
        data[key]['chunkedsignature'] = np.array_split(data[key]['signature'], b)


    # We turn the numbers in a band into a string which will help with our implementation of locality sensitive hashing
    for key in bootStrapTestList:
        data[key]['chunkedsignature_asStrings'] = list()
        for band in range(b):
            intermediateList = [str(i) for i in data[key]['chunkedsignature'][band]]
            string_band = int(''.join(intermediateList))
            data[key]['chunkedsignature_asStrings'].append(string_band)

    # NOW WE PERFORM LSH AND RETRIEVE THE CANDIDATE PAIRS
    # now we will put the chunks into buckets.
    # Note the number of buckets are chosen larger than the number of products in order to avoid collisions if the bands are not exactly the same
    candidateset = set()
    for band in range(b):
        bandDictionary = dict()
        for key in bootStrapTestList:
            if data[key]['chunkedsignature_asStrings'][band] in bandDictionary.keys():
                bandDictionary[data[key]['chunkedsignature_asStrings'][band]].append(key)

            else:
                bandDictionary[data[key]['chunkedsignature_asStrings'][band]] = list()
                bandDictionary[data[key]['chunkedsignature_asStrings'][band]].append(key)

        for bucketKeys in bandDictionary:
            for indexA, a in enumerate(bandDictionary[bucketKeys]):
                for indexB in range(indexA + 1, len(bandDictionary[bucketKeys])):
                    candidateset.add((a, bandDictionary[bucketKeys][indexB]))

    # Now we turn our candidate set to a list
    candidateList = list(candidateset)
    for index, tup in enumerate(candidateList):
        candidateList[index] = tuple(sorted(tup))
    candidateset = set(candidateList)
    candidateList = list(candidateset)

    # NOW WE COMPUTE FOR OUR BOOTSTRAP:
    # - Fraction of comparisons made
    # - Pair Quality
    # - Pair completeness
    # - F1* measure
    totalNumberOfPossibleComparisons = comb(len(bootStrapTestList), 2)
    numberOfComparisonsMAde = len(candidateList)
    fractionOfComparisonMade = numberOfComparisonsMAde / totalNumberOfPossibleComparisons

    # FIRST I FIND THE TOTAL NUMBER OF DUPLICATES IN THIS TEST SET:
    # - First i find all possible pairs
    # - Then i look how many of those are duplicates
    allPossiblePairs = set()
    totalAmountOfDuplicates = 0
    for indexA, a in enumerate(bootStrapTestList):
        for indexB in range(indexA + 1, len(bootStrapTestList)):
            allPossiblePairs.add((a, bootStrapTestList[indexB]))

    for pair in allPossiblePairs:
        if data[pair[0]]['modelID'] == data[pair[1]]['modelID']:
            totalAmountOfDuplicates = totalAmountOfDuplicates + 1

    # now i compute the remainder of the things we need for pairs quality & pair completeness
    numberOfDuplicatesFound = 0

    for candidates in candidateList:
        if data[candidates[0]]['modelID'] == data[candidates[1]]['modelID']:
            numberOfDuplicatesFound = numberOfDuplicatesFound + 1
    numberOfComparisonsMAde = len(candidateList)
    pairQuality = numberOfDuplicatesFound / numberOfComparisonsMAde
    pairCompleteness = numberOfDuplicatesFound / totalAmountOfDuplicates
    if (pairQuality != 0 or pairCompleteness != 0):
        f1_star = (2 * pairQuality * pairCompleteness) / (pairQuality + pairCompleteness)
    else:
        f1_star = 0

    # NOW WE CREATE THE DISSIMILARITY MATRIX
    gamma = tunedParameters[2]
    dissimilarityMatrix = np.asarray(np.ones((len(bootStrapTestList), len(bootStrapTestList))) * 1000000)

    for candidates in candidateList:
        if ((data[candidates[0]]['featuresMap']['Brand'] != data[candidates[1]]['featuresMap']['Brand']) and
                data[candidates[0]]['featuresMap']['Brand'] != 'nobrand' and data[candidates[1]]['featuresMap'][
                    'Brand'] != 'nobrand'):

            dissimilarityMatrix[productToMatrixMapper[candidates[0]], productToMatrixMapper[candidates[1]]] = 1000000

        elif data[candidates[0]]['shop'] == data[candidates[1]]['shop']:
            dissimilarityMatrix[productToMatrixMapper[candidates[0]], productToMatrixMapper[candidates[1]]] = 1000000

        else:
            sim = 0
            avgSim = 0
            m = 0
            w = 0
            nonMatchingKeysA = set(data[candidates[0]]['featuresMap'].keys())
            nonMatchingKeysB = set(data[candidates[1]]['featuresMap'].keys())
            for keysA in data[candidates[0]]['featuresMap']:
                for keysB in data[candidates[1]]['featuresMap']:
                    keySim = stringSimilarity(keysA, keysB)
                    if keySim > gamma:
                        valueSim = stringSimilarity(data[candidates[0]]['featuresMap'][keysA],
                                                    data[candidates[1]]['featuresMap'][keysB])
                        weight = keySim
                        sim = sim + weight * valueSim
                        m = m + 1
                        w = w + weight
                        nonMatchingKeysA.remove(keysA)
                        nonMatchingKeysB.remove(keysB)

            if w > 0:
                avgSim = sim / w

            setModelWordsA = set()
            setModelWordsB = set()
            for keyz1 in nonMatchingKeysA:
                for bla in extractModelWordsFromString(data[candidates[0]]['featuresMap'][keyz1]):
                    setModelWordsA.add(bla)

            for keyz2 in nonMatchingKeysB:
                for bla2 in extractModelWordsFromString(data[candidates[1]]['featuresMap'][keyz2]):
                    setModelWordsB.add(bla2)

            intersectionBoth = setModelWordsA.intersection(setModelWordsB)
            if len(setModelWordsA) > 0 and len(setModelWordsB) > 0:
                mwPerc = (len(intersectionBoth) * 2) / (len(setModelWordsA) + len(setModelWordsB))
            else:
                mwPerc = 0

            titleSim = titleModelWordsDistance(data[candidates[0]]['title'], data[candidates[1]]['title'],tunedParameters[4],tunedParameters[5],tunedParameters[3],tunedParameters[6])

            if titleSim == -1:
                theta1 = m / min(len(data[candidates[0]]['featuresMap'].keys()),
                                 len(data[candidates[1]]['featuresMap'].keys()))
                theta2 = 1 - theta1
                hSIM = theta1 * avgSim + theta2 * mwPerc
                dist = 1 - hSIM

            else:
                theta1 = (1 - tunedParameters[0]) * (m / min(len(data[candidates[0]]['featuresMap'].keys()),
                                             len(data[candidates[1]]['featuresMap'].keys())))
                theta2 = 1 - tunedParameters[0] - theta1
                hSIM = theta1 * avgSim + theta2 * mwPerc + tunedParameters[0] * titleSim
                dist = 1 - hSIM

            dissimilarityMatrix[productToMatrixMapper[candidates[0]], productToMatrixMapper[candidates[1]]] = dist

    # Now i make sure the other symmetric distances are also filled in.
    for i in range(dissimilarityMatrix.shape[0]):
        for j in range(i + 1, dissimilarityMatrix.shape[0]):
            if not dissimilarityMatrix[i, j] == 1000000:
                dissimilarityMatrix[j, i] = dissimilarityMatrix[i, j]

    # NOW WE PERFORM THE CLUSTERING FOR ALL DISSIMILARITY MATRICES AND CALCULATE THE F1 SCORE
    dictionaryWithClusters = performClustering(dissimilarityMatrix, matrixToProductMapper, tunedParameters[1])
    listWithPredictedPairs = getPredictedPairsFromClustering(dictionaryWithClusters)

    TruePositives = 0
    FalsePositives = 0
    FalseNegatives = 0

    for element in listWithPredictedPairs:
        if data[element[0]]['modelID'] == data[element[1]]['modelID']:
            TruePositives = TruePositives + 1
        else:
            FalsePositives = FalsePositives + 1

    allPossiblePairsClust = set()
    for index1, element in enumerate(bootStrapTestList):
        for index2 in range(index1 + 1, len(bootStrapTestList)):
            allPossiblePairsClust.add((element, bootStrapTestList[index2]))

    predictedToBeNegative = allPossiblePairsClust.difference(set(listWithPredictedPairs))
    predictedToBeNegative = list(predictedToBeNegative)

    for pred in predictedToBeNegative:
        if data[pred[0]]['modelID'] == data[pred[1]]['modelID']:
            FalseNegatives = FalseNegatives + 1

    if TruePositives > 0 or FalsePositives > 0:
        precision = TruePositives / (TruePositives + FalsePositives)
    else:
        precision = 0

    if TruePositives > 0 or FalseNegatives > 0:
        recall = TruePositives / (TruePositives + FalseNegatives)
    else:
        recall = 0

    # NOTE SOMETIMES WE GET 0 CANDIDATES!
    if precision > 0 or recall > 0:
        F1 = (2 * precision * recall) / (precision + recall)
    else:
        F1 = 0

    return [fractionOfComparisonMade,pairQuality,pairCompleteness,f1_star,F1,precision ,recall]

def perform5Bootstraps_andReturnAveragePerformanceMeasures(t):
    bootstrap1 =performOneBootstrap(t)
    bootstrap2 =performOneBootstrap(t)
    bootstrap3 =performOneBootstrap(t)
    bootstrap4 =performOneBootstrap(t)
    bootstrap5 =performOneBootstrap(t)

    averageFractionOfComparisonsMade = (bootstrap1[0] + bootstrap2[0] + bootstrap3[0] + bootstrap4[0] + bootstrap5[0])/5
    averagePairQuality = (bootstrap1[1] + bootstrap2[1] + bootstrap3[1] + bootstrap4[1] + bootstrap5[1])/5
    averagePairCompleteness = (bootstrap1[2] + bootstrap2[2] + bootstrap3[2] + bootstrap4[2] + bootstrap5[2])/5
    averagef1_star = (bootstrap1[3] + bootstrap2[3] + bootstrap3[3] + bootstrap4[3] + bootstrap5[3])/5
    averagef1 = (bootstrap1[4] + bootstrap2[4] + bootstrap3[4] + bootstrap4[4] + bootstrap5[4])/5
    averagePrecision = (bootstrap1[5] + bootstrap2[5] + bootstrap3[5] + bootstrap4[5] + bootstrap5[5])/5
    averageRecall = (bootstrap1[6] + bootstrap2[6] + bootstrap3[6] + bootstrap4[6] + bootstrap5[6])/5

    return [averageFractionOfComparisonsMade,averagePrecision,averageRecall,averagePairQuality,averagePairCompleteness,averagef1_star,averagef1]



def plotPerformanceAcrossMultipleThresholds():
    threshold1 = perform5Bootstraps_andReturnAveragePerformanceMeasures(0.3)
    threshold2 = perform5Bootstraps_andReturnAveragePerformanceMeasures(0.4)
    threshold3 = perform5Bootstraps_andReturnAveragePerformanceMeasures(0.5)
    threshold4 = perform5Bootstraps_andReturnAveragePerformanceMeasures(0.6)
    threshold5 = perform5Bootstraps_andReturnAveragePerformanceMeasures(0.65)
    threshold6 = perform5Bootstraps_andReturnAveragePerformanceMeasures(0.7)
    threshold7 = perform5Bootstraps_andReturnAveragePerformanceMeasures(0.75)
    print(threshold1)
    print(threshold2)
    print(threshold3)
    print(threshold4)
    print(threshold5)
    print(threshold6)
    print(threshold7)
    thresholdList = list()
    thresholdList.append(threshold1)
    thresholdList.append(threshold2)
    thresholdList.append(threshold3)
    thresholdList.append(threshold4)
    thresholdList.append(threshold5)
    thresholdList.append(threshold6)
    thresholdList.append(threshold7)
    fractionOfComparisonsList = list()
    precisionList = list()
    recallList = list()
    pairQualityList = list()
    pairCompletenessList = list()
    f1starList = list()
    f1List = list()
    for lists in thresholdList:
        fractionOfComparisonsList.append(lists[0])
        precisionList.append(lists[1])
        recallList.append(lists[2])
        pairQualityList.append(lists[3])
        pairCompletenessList.append(lists[4])
        f1starList.append(lists[5])
        f1List.append(lists[6])

    plt.plot(fractionOfComparisonsList,pairQualityList)
    plt.xlabel('fraction of comparisons')
    plt.ylabel('pair quality')
    plt.show()

    plt.plot(fractionOfComparisonsList, pairCompletenessList)
    plt.xlabel('fraction of comparisons')
    plt.ylabel('pair completeness')
    plt.show()

    plt.plot(fractionOfComparisonsList, f1starList)
    plt.xlabel('fraction of comparisons')
    plt.ylabel('f1*')
    plt.show()

    plt.plot(fractionOfComparisonsList, f1List)
    plt.xlabel('fraction of comparisons')
    plt.ylabel('f1')
    plt.show()

    plt.plot(fractionOfComparisonsList, recallList)
    plt.xlabel('fraction of comparisons')
    plt.ylabel('recall')
    plt.show()

    plt.plot(fractionOfComparisonsList, precisionList)
    plt.xlabel('fraction of comparisons')
    plt.ylabel('precision')
    plt.show()




plotPerformanceAcrossMultipleThresholds()
#perform5Bootstraps_andReturnAveragePerformanceMeasures(0.65)


# added dimensions to featuresToConsider
# average linkage used
# featuresToConsider = ['size', 'width', 'height', 'depth', 'ratio', 'weight', 'resolution', 'refresh']

