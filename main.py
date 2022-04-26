# -*- coding: utf-8 -*-
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import napoli
import popularity
from sklearn.utils import parallel_backend
from sklearn.model_selection import cross_val_score
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as fn

#import ALS_custom


def main(spark, in_path, out_path):
    '''
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('LOADING....')
    print('')

	# TODO will have to change implementation of napoliSplit if we want a terminal written for in_path argument --> edit readRDD.py helper function
    print('Splitting the ratings dataset into training, validation and test set')
    ratings_train, ratings_test, ratings_validation = napoli.napoliSplit(spark, in_path, small=True, column_name = 'ratings', prop = 0.8)
    ratings_train.show()


    # split into training and testing sets
    #ratings_train.write.csv(f'{out_path}/ratings_train.csv')
    #ratings_validation.write.csv(f'{out_path}/ratings_validation.csv')
    #ratings_test.write.csv(f'{out_path}/ratings_test.csv')

    print("Distinct movies: ", ratings_train.select("movieId").distinct().count())
    print("Distinct users: ", ratings_train.select("userId").distinct().count())
    print("Total number of ratings: ", ratings_train.count())

    ratings_per_user = ratings_train.groupby('userId').agg({"rating":"count"})
    ratings_per_user.describe().show()

    ratings_per_movie = ratings_train.groupby('movieId').agg({"rating":"count"})
    ratings_per_movie.describe().show()

    X_train, X_test, X_val = ratings_train.drop('timestamp'), ratings_test.drop('timestamp'), ratings_validation.drop('timestamp')
    print("Training data size : ", X_train.count())
    print("Validation data size : ", X_val.count())
    print("Test data size : ", X_test.count())
    print("Distinct users in Training set : ", X_train[["userId"]].distinct().count())
    print("Distinct users in Test set : ", X_test[["userId"]].distinct().count())
    print("Distinct users in Validation set: ", X_val[["userId"]].distinct().count())

    print("Fitting Popularity baseline model")
    print("Tuning hyperparameters based on Mean Average Precision")
    damping_values = [0, 5, 10, 15, 20]
    best_score = 0
    best_model = None
    for damping in damping_values:
        baseline = PopularityBaseline(damping = damping)
        baseline.fit(X_train)
        baseline_metrics_val = baseline.evaluate(baseline.results, X_val)
        val_score = baseline_metrics_val.meanAveragePrecision
        if val_score > best_score:
            best_score = val_score
            best_baseline_model = baseline
    print('Best Popularity baseline model: ', best_baseline_model)

    print("Evaluating best Popularity baseline model")
    baseline_metrics_train = baseline.evaluate(best_baseline_model.results, X_train)
    baseline_metrics_test = baseline.evaluate(best_baseline_model.results, X_test)
    print("Mean Average Precision on training set: ", baseline_metrics.meanAveragePrecision)
    print("Mean Average Precision on test set: ", baseline_metrics.meanAveragePrecision)


    print("Fitting Latent Factor model with ALS")
    als = ALS(userCol="userId",itemCol="movieId",ratingCol="rating",rank=5, maxIter=10, seed=0)
    model = als.fit(X_train)

    # displaying the latent features for 10 users
    model.userFactors.show(10, truncate = False)

    # See users and/or items in the validation dataset that were not part of the training dataset and transform() method implementation of ALS returns NaN for their predictions
    model.transform(X_val).where(fn.isnan('prediction')).show(5)
    model.transform(X_val[['userId','movieId']]).na.drop()[['prediction']].show()

    print('Evaluating Latent Factor model')
    # establish evaluation metric
    eval=RegressionEvaluator(metricName="rmse",labelCol="rating", predictionCol="prediction")

    # check baseline ALS model performancee
    train_predictions = model.transform(X_train)
    test_predictions = model.transform(X_test).na.drop()
    print("Base RMSE on training data : ", eval.evaluate(train_predictions))
    print("Base RMSE on test data: ", eval.evaluate(test_predictions))



    # Build the recommendation model using ALS on the training data. Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    #als_model = ALS_custom.alsDF(ratings_train, ratings_test, maxIter=5, userCol="userId", itemCol="movieId", ratingCol="rating")

    # Fit the ALS model to the training set

    # Tune params against validation data; evaluate the model by computing the RMSE (?) on the test data

    #  initialize the ALS model

    # create the parameter grid

    # instantiating crossvalidator estimator

    # now incorporate names of movies to people, essentialy: user xyz would like movieId 123 "abc", in genre tag "a|b|c" in a data structure





# Only enter this block if we're in main
if __name__ == "__main__":

    spark = SparkSession.builder.appName('project')\
        .config('spark.submit.pyFiles', 'Group26_MovieLens-0.1.0-py3-none-any.zip')\
        .config('spark.shuffle.useOldFetchProtocol', 'true')\
        .config('spark.shuffle.service.enabled','true')\
        .config('dynamicAllocation.enabled', 'true')\
        .getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

	# Get path of ratings file
    in_path = f'hdfs:/user/{netID}' #sys.argv[1]

    # Get destination directory of training, validation and test set files
    out_path = f'hdfs:/user/{netID}' #sys.argv[2]

    # Call our main routine
    main(spark, in_path, out_path)
