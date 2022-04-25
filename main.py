#Use getpass to obtain user netID
import getpass
from os import truncate
from statistics import median
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import napoli


def main(spark, in_path, out_path):
    '''
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('LOADING....')
	# TODO will have to change implementation of napoliSplit if we want a terminal written for in_path argument --> edit readRDD.py helper function
    ratings_train, ratings_test, ratings_validation = napoli.napoliSplit(in_path, small=True, column_name = 'ratings', upper_prop = 0.9, lower_prop = 0.8)

    ratings_train.write.csv(f'{out_path}/ratings_train.csv')
    ratings_validation.write.csv(f'{out_path}/ratings_validation.csv')
    ratings_test.write.csv(f'{out_path}/ratings_test.csv')

   
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('project').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

	# Get path of ratings file
    in_path = f'hdfs:/user/{netID}' #sys.argv[1]
    
    # Get destination directory of training, validation and test set files
    out_path = f'hdfs:/user/{netID}' #sys.argv[2]

    # Call our main routine
    main(spark, in_path, out_path)