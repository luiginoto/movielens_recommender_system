# -*- coding: utf-8 -*- 
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import napoli
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
    ratings_train, ratings_test, ratings_validation = napoli.napoliSplit(spark, in_path, small=False, column_name = 'ratings', prop = 0.8)

    #ratings_train.write.csv(f'{out_path}/ratings_train.csv')
    #ratings_validation.write.csv(f'{out_path}/ratings_validation.csv')
    #ratings_test.write.csv(f'{out_path}/ratings_test.csv')

    #als_model = ALS_custom.alsDF(ratings_train, ratings_test, maxIter=5, userCol="userId", itemCol="movieId", ratingCol="rating")

   
    
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