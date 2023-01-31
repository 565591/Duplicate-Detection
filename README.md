# Duplicate-Detection
In this project I use LSH and agglomerative clustering to find duplicate products across web shops.

Using the function "plotPerformanceAcrossMultipleThresholds()" will run 5 bootstraps for each of the following LSH thresholds: 0.3, 0.4, 0.5,0.6,0.65,0.7,0.75 and record the average performance across the 5 bootstraps and then make plots of all the performance measures with respect to the amount of comparisons made as a percentage of total amount of possible comparisons.

Using the function "perform5Bootstraps_andReturnAveragePerformanceMeasures(t)" will run 5 bootstraps for the threshold "t" that needs to be given as input and returns the average performance measures across the 5 bootstraps in the following order: averageFractionOfComparisonsMade,averagePrecision,averageRecall,averagePairQuality,averagePairCompleteness,averagef1_star,averagef1
