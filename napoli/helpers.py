
import warnings
from pyspark.sql.window import Window
from pyspark.sql import functions as fn

def readRDD(dirstring, small, column_name):
    if small:
        dir = dirstring + '/ml-latest-small/'
        column_names = ['movies','ratings','links','tags']
        # Load into DataFrame
        if column_name in column_names:
            if column_name == 'movies':
               return spark.read.csv(dir + 'movies.csv', header=True, schema='movieId INT, title STRING, genres STRING')
            elif column_name == 'links':
                return spark.read.csv(dir + 'links.csv', header=True, schema='movieId INT, imdbId FLOAT, tmdbId FLOAT')
            elif column_name == 'ratings':
                return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
            elif column_name == 'tags':
                return spark.read.csv(dir + 'tags.csv', header=True, schema='userId INT, movieId INT, tag STRING, timestamp INT'), column_name
        else:
            warnings.warn('Warning Message: Column name not found; opting for ratings')
            return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), 'ratings'
    else:
        dir = dirstring + '/ml-latest/'
        column_names = ['movies','ratings','links','tags','genome-tags','genome-scores']
        # Load into DataFrame
        if column_name in column_names:
            if column_name == 'movies':
                return spark.read.csv(dir + 'movies.csv', header=True, schema='movieId INT, title STRING, genres STRING')
            elif column_name == 'links':
                return spark.read.csv(dir + 'links.csv', header=True, schema='movieId INT, imdbId FLOAT, tmdbId FLOAT')
            elif column_name == 'ratings':
                return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
            elif column_name == 'tags':
                return spark.read.csv(dir + 'tags.csv', header=True, schema='userId INT, movieId INT, tag STRING, timestamp INT')
            elif column_name == 'genome-scores':
                return spark.read.csv(dir + 'genome-scores.csv', header=True, schema='movieId INT, tagId INT, relevance STRING')
            elif column_name == 'genome-tags':
                return spark.read.csv(dir + 'genome-tags.csv', header=True, schema='tagId INT, tag STRING'), column_name
        else:
            warnings.warn('Warning Message: Column name not found; opting for ratings')
            return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), 'ratings'

def ratings_split(rdd, upper_lim, lower_lim):
    windowSpec  = Window.partitionBy('userId').orderBy('timestamp')
    
    ratings = rdd \
            .withColumn('row_number', fn.row_number().over(windowSpec)) \
            .withColumn('n_ratings', fn.count('rating').over(windowSpec)) \
            .withColumn('prop_idx', (fn.col('row_number') / fn.col('n_ratings')))
    ratings.show()

    ratings_train = ratings.filter(ratings.prop_idx <= lower_lim)
    ratings_validation = ratings.filter((ratings.prop_idx > lower_lim) & (ratings.prop_idx <= upper_lim)) 
    ratings_test = ratings.filter((ratings.prop_idx > upper_lim) & (ratings.prop_idx <= 1.0))

    return ratings_train, ratings_test, ratings_validation

