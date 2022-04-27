# -*- coding: utf-8 -*-
#Use getpass to obtain user netID
import getpass
from dataset_split.helpers import readRDD

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import popularity
import dataset_split
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as fn
from pyspark import SparkConf, SparkContext

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
    ratings_train, ratings_test, ratings_validation = dataset_split.ratingsSplit(spark, in_path, small=True, column_name = 'ratings', prop = 0.8)
    ratings_train.show()

    movie_title_df, _ = readRDD(spark, in_path, small=True, column_name = 'movies')


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
    best_baseline_model = None
    for damping in damping_values:
        baseline = popularity.PopularityBaseline(damping = damping)
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
    print("Mean Average Precision on training set: ", baseline_metrics_train.meanAveragePrecision)
    print("Mean Average Precision on test set: ", baseline_metrics_test.meanAveragePrecision)

    print("Fitting Latent Factor model with ALS")
    als = ALS(userCol="userId",itemCol="movieId",ratingCol="rating",rank=5, maxIter=10, coldStartStrategy="nan", seed=0)
    model = als.fit(X_train)

    # displaying the latent features for 10 users
    #model.userFactors.show(10, truncate = False)

    # Get predicted ratings on all existing user-movie pairs
    predictions = model.transform(ratings_validation).drop('timestamp')
    predictions.show()

    # Get predicted ratings on all existing user-movie pairs
    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RankingEvaluator.html#pyspark.ml.evaluation.RankingEvaluator

    # Note the evaluator ingests pyspark dataframe NOT rdd
    df_label = predictions.groupBy('userId').agg(fn.collect_list('movieId').alias('label'))
    df_pred = predictions.groupBy('userId').agg(fn.collect_list('prediction').alias('prediction'))
    data = df_label.join(df_pred, ['userId'])
    data.rdd.map(tuple)
    #data.show()
    sc = SparkContext().getOrCreate()
    metrics = RankingMetrics(data)
    print("ALS MAP@100 on validation set: ", metrics.meanAveragePrecisionAt(100))

    #num_recs = 10
    #user_id = 100
    #recommendations = model.recommendForAllUsers(num_recs)

    #def name_retriever(movie_id, movie_title_df):
    #    return movie_title_df.where(movie_title_df.movieId == movie_id).take(1)[0]['title']
    
    # get recommendations specifically for the new user that has been added to the DataFrame
    #recs_for_user = recommendations.where(recommendations.userId == user_id).take(1)
    
    #for ranking, (movie_id, rating) in enumerate(recs_for_user[0]['recommendations']):
    #    movie_string = name_retriever(movie_id, movie_title_df)
    #    print('Recommendation {}: {}  | predicted score :{}'.format(ranking+1,movie_string,rating))
        
    


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
