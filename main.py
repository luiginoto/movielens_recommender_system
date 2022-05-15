# -*- coding: utf-8 -*-
# Use getpass to obtain user netID
import getpass
from validated_models.ALS import CustomALS
from dataset_split.utils import readRDD
from validated_models.popularity import PopularityBaseline
from validated_models.popularity_validation import PopularityBaselineValidation
from validated_models.als_validation import ALSValidation
import pyspark.sql.functions as fn
from pyspark.ml.recommendation import ALS


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import dataset_split


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
    
    seed = 69

    print('Splitting the ratings dataset into training, validation and test set')
    ratings_train, ratings_test, ratings_validation = dataset_split.ratingsSplit(
        spark, in_path, small=False, column_name='ratings', train_ratio=0.8, user_ratio=0.5, seed =seed)
    ratings_train.show()

    #movie_title_df, _ = readRDD(spark, in_path, small=True, column_name = 'movies')

    # split into training and testing sets
    # ratings_train.write.csv(f'{out_path}/ratings_train.csv')
    # ratings_validation.write.csv(f'{out_path}/ratings_validation.csv')
    # ratings_test.write.csv(f'{out_path}/ratings_test.csv')

    print("Distinct movies: ", ratings_train.select(
        "movieId").distinct().count())
    print("Distinct users: ", ratings_train.select("userId").distinct().count())
    print("Total number of ratings: ", ratings_train.count())

    X_train, X_test, X_val = ratings_train.drop('timestamp'), ratings_test.drop(
        'timestamp'), ratings_validation.drop('timestamp')
    
    ratings_per_user = ratings_train.groupby('userId').agg({"rating":"count"})
    ratings_per_user.describe().show()

    ratings_per_movie = ratings_train.groupby('movieId').agg({"rating":"count"})
    ratings_per_movie.describe().show()
 
    print("Training data size : ", X_train.count())
    print("Validation data size : ", X_val.count())
    print("Test data size : ", X_test.count())
    print("Distinct users in Training set : ", X_train[["userId"]].distinct().count())
    print("Distinct users in Test set : ", X_test[["userId"]].distinct().count())
    print("Distinct users in Validation set: ", X_val[["userId"]].distinct().count())
    
    print("Fitting Popularity baseline model")
    print("Tuning hyperparameters based on Mean Average Precision")
    damping_values = [0, 5, 10, 15, 20]
    best_baseline_model = PopularityBaselineValidation(X_train, X_val, damping_values)
    print("Evaluating best Popularity baseline model")
    baseline_metrics_train = best_baseline_model.evaluate(best_baseline_model.results, X_train)
    baseline_metrics_test = best_baseline_model.evaluate(best_baseline_model.results, X_test)
    print("MAP@100 on training set: ", baseline_metrics_train.meanAveragePrecision)
    print("MAP@100 on test set: ", baseline_metrics_test.meanAveragePrecision)
    print("NCDG@100 on training set: ", baseline_metrics_train.ndcgAt(100))
    print("NCDG@100 on test set: ", baseline_metrics_test.ndcgAt(100))
    best_popularity_scores = best_baseline_model.popularity
    
    print("Fitting ALS model")
    print("Tuning hyperparameters based on Mean Average Precision")
    
    rank_values = [30, 40, 50]
    regParam_values = [0.001, 0.01, 0.1]
    maxIter_values = [20, 22, 26, 28]
  
    
    best_als_model = ALSValidation(X_train, X_val, rank_vals=rank_values, regParam_vals=regParam_values, maxIter_vals=maxIter_values)
    
    print("Evaluating best ALS model")
    als_metrics_train = best_als_model.evaluate(X_train)
    als_metrics_test = best_als_model.evaluate(X_test)
    print("MAP@100 on training set: ", als_metrics_train.meanAveragePrecision)
    print("MAP@100 on test set: ", als_metrics_test.meanAveragePrecision)
    print("NCDG@100 on training set: ", als_metrics_train.ndcgAt(100))
    print("NCDG@100 on test set: ", als_metrics_test.ndcgAt(100))
    best_user_factors = best_als_model.fitted_model.userFactors
    best_item_factors = best_als_model.fitted_model.itemFactors
    best_predsAndlabels = best_als_model.predsAndlabels
    
    print()
    print('Exporting training ratings')
    ratings_train.repartition(1).write.csv(out_path + '/final_model_results/ratings_train.csv', mode='overwrite')
    
    print()
    print('Exporting Popularity scores of best Popularity baseline model into CSV')
    
    best_popularity_scores.repartition(1).write.csv(out_path + '/final_model_results/popularity_scores.csv', mode='overwrite')
    
    print()
    print('Exporting User and Item factors of best ALS model into CSV')
    
    best_user_factors.withColumn("features", best_user_factors.features.cast("array<string>"))\
        .withColumn("features", fn.concat_ws(",",fn.col("features")))\
        .repartition(1).write.csv(out_path + '/final_model_results/user_factors.csv', mode='overwrite')
        
    best_item_factors.withColumn("features", best_item_factors.features.cast("array<string>"))\
        .withColumn("features", fn.concat_ws(",",fn.col("features")))\
        .repartition(1).write.csv(out_path + '/final_model_results/item_factors.csv', mode='overwrite')
        
    print()
    print('Exporting Predictions of best ALS model with test Ground Truth labels for each user into CSV')
    
    best_predsAndlabels.withColumn("recommendations", fn.concat_ws(",",fn.col("recommendations")))\
        .withColumn("liked_movies", fn.concat_ws(",",fn.col("liked_movies")))\
        .repartition(1).write.csv(out_path + '/final_model_results/predsAndlabels.csv', mode='overwrite')
    


# Only enter this block if we're in main
if __name__ == "__main__":

    spark = SparkSession.builder.appName('project')\
        .config('spark.submit.pyFiles', 'Group26_MovieLens-0.5.0-py3-none-any.zip')\
        .config('spark.shuffle.useOldFetchProtocol', 'true')\
        .config('spark.shuffle.service.enabled', 'true')\
        .config('dynamicAllocation.enabled', 'true')\
        .config('spark.task.maxFailures', '2')\
        .getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Get path of ratings file
    in_path = f'hdfs:/user/{netID}'  # sys.argv[1]

    # Get destination directory of training, validation and test set files
    out_path = f'hdfs:/user/{netID}'  # sys.argv[2]

    # Call our main routine
    main(spark, in_path, out_path)
