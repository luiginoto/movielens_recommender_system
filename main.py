# -*- coding: utf-8 -*- 
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import data_split
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml.recommendation import ALS 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

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
    ratings_train, ratings_test, ratings_validation = data_split.train_test_val(spark, in_path, small=True, column_name = 'ratings', prop = 0.8)

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

    # See users and/or items in the validation dataset that were not part of the training dataset and transform() method implementation of ALS returns NaN for their predictions
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics, else join may ltaer take care of this 
    als = ALS(userCol="userId",itemCol="movieId",ratingCol="rating",rank=5, maxIter=10, seed=0, coldStartStrategy="drop")
    model = als.fit(X_train)

    # Get predicted ratings on all existing user-movie pairs
    predictions = model.transform(ratings_validation).drop('timestamp')
    predictions.show()

    # Get predicted ratings on all existing user-movie pairs
    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RankingEvaluator.html#pyspark.ml.evaluation.RankingEvaluator

    # Note the evaluator ingests pyspark dataframe NOT rdd
    df_label = predictions.groupBy('movieId').agg(F.collect_list('rating').alias('label_bad_float'))
    df_pred = predictions.groupBy('movieId').agg(F.collect_list('prediction').alias('prediction_bad_float'))
    raw = df_label.join(df_pred, ['movieId'])

     # Note the evaluator ingests pyspark dataframe NOT rdd
    #df = VectorAssembler(inputCols=['label_bad_float'], outputCol='label').transform(raw)
    #df = VectorAssembler(inputCols=['userId','movieId'], outputCol='label').transform(df)
    #df = df.select(vector_to_array('label').alias('label'))

    evaluator = RankingEvaluator.s

    # is this the right metric -- see documentation and requirements
    evaluator.evaluate(raw, {evaluator.metricName: "precisionAtK", evaluator.k: 100})


 


   
    
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