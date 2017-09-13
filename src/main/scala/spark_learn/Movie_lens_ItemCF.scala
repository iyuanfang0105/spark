package spark_learn

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

object Movie_lens_ItemCF {


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val sparkConf = new SparkConf()
    sparkConf.setAppName("recommend film sys")
    sparkConf.setMaster("local[2]")

    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sparkContext)


    val PRIOR_COUNT = 10
    val PRIOR_CORRELATION = 0

    // load the movies info, format: (movieId, movieName)
    val movie_lens_data_path = "/home/meizu/WORK/public_dataset/movie_lens/ml-100k"
    val movies: RDD[(Int, String)] = sparkContext.textFile(movie_lens_data_path + "/u.item").map { line =>
      val fields = line.split("\\|")
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }
    val movieNames = movies.collectAsMap()
    println("\n\n**************** Movies: *************** \n\n")
    movies.take(10).foreach(println)

    // load rating info, format: (userId, movieId, rating)
    val ratings: RDD[(Int, Int, Double)] = sparkContext.textFile(movie_lens_data_path + "/u.data").map { line =>
      val fields = line.split("\t")
      // format: (userId, movieId, rating)
      (fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }

    println("\n\n**************** Ratings 10 samples: *************** \n\n")
    ratings.take(10).foreach(println)
    val rating_num = ratings.count()
    val user_num = ratings.map(_._1).distinct.count
    val movie_num = ratings.map(_._2).distinct.count
    println("\n====>>>>  Ratings Counts: " + rating_num)
    println("====>>>>  User Counts: " + user_num)
    println("====>>>>  Movie Counts: " + movie_num)

    // get num raters per movie, keyed on movie id, format: (movieId, num_raters)
    val raters_per_movie_num: RDD[(Int, Double)] = ratings.groupBy(_._2).map(grouped => (grouped._1, grouped._2.size))

    // join ratings with num raters on movie id, format: (userId, movieId, rating, num_raters)
    // RDD[(Int, (Iterable[(Int, Int, Double)], Double))] >>> (userId, movieId, rating, num_raters)
    val ratings_with_size: RDD[(Int, Int, Double, Double)] = ratings.groupBy(_._2).join(raters_per_movie_num).flatMap(joined => {
      joined._2._1.map(f => (f._1, f._2, f._3, joined._2._2))
    })
    ratings_with_size.take(10).foreach(println)

    // dummy copy of ratings for self join,
    // format: (userId, (userId, movieId, rating, num_raters))
    val ratings2 = ratings_with_size.keyBy(v => v._1)
    // join on userid and filter movie pairs such that we don't double-count and exclude self-pairs
    val rating_pairs: RDD[(Int, ((Int, Int, Double, Double), (Int, Int, Double, Double)))] = ratings_with_size.keyBy(tup => tup._1).join(ratings2)
        .filter(f => f._2._1._2 < f._2._2._2)

    // compute raw inputs to similarity metrics for each movie pair
    val vectorCalcs =rating_pairs
        .map(data => {
          val key = (data._2._1._2, data._2._2._2) // movieId1, movieId2
          val stats =
            (data._2._1._3 * data._2._2._3, // rating 1 * rating 2
              data._2._1._3,                // rating movie 1
              data._2._2._3,                // rating movie 2
              math.pow(data._2._1._3, 2),   // square of rating movie 1
              math.pow(data._2._2._3, 2),   // square of rating movie 2
              data._2._1._4,                // number of raters movie 1
              data._2._2._4)                // number of raters movie 2
          (key, stats)
        })
        .groupByKey()
        .map(data => {
          val key = data._1
          val vals = data._2
          val size = vals.size
          val dotProduct = vals.map(f => f._1).sum
          val ratingSum = vals.map(f => f._2).sum
          val rating2Sum = vals.map(f => f._3).sum
          val ratingSq = vals.map(f => f._4).sum
          val rating2Sq = vals.map(f => f._5).sum
          val numRaters = vals.map(f => f._6).max
          val numRaters2 = vals.map(f => f._7).max
          (key, (size, dotProduct, ratingSum, rating2Sum, ratingSq, rating2Sq, numRaters, numRaters2))
        })
    // compute similarity metrics for each movie pair
    val similarities = vectorCalcs
        .map(fields => {
          val key = fields._1
          val (size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq, numRaters, numRaters2) = fields._2
          val corr = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
          val regCorr = regularizedCorrelation(size, dotProduct, ratingSum, rating2Sum,
            ratingNormSq, rating2NormSq, PRIOR_COUNT, PRIOR_CORRELATION)
          val cosSim = cosineSimilarity(dotProduct, scala.math.sqrt(ratingNormSq), scala.math.sqrt(rating2NormSq))
          val jaccard = jaccardSimilarity(size, numRaters, numRaters2)

          (key, (corr, regCorr, cosSim, jaccard))
        })
    println("\n====>>>> Similarity Matrix: ")
    similarities.take(10).foreach(println)

    // test a few movies out (substitute the contains call with the relevant movie name
    val sample = similarities.filter(m => {
      val movies = m._1
      (movieNames(movies._1).contains("Star Wars (1977)"))
    })

    // collect results, excluding NaNs if applicable
    val result = sample.map(v => {
      val m1 = v._1._1
      val m2 = v._1._2
      val corr = v._2._1
      val rcorr = v._2._2
      val cos = v._2._3
      val j = v._2._4
      (movieNames(m1), movieNames(m2), corr, rcorr, cos, j)
    }).collect().filter(e => !(e._4 equals Double.NaN))    // test for NaNs must use equals rather than ==
      .sortBy(elem => elem._4).take(10)

    // print the top 10 out
    result.foreach(r => println(r._1 + " | " + r._2 + " | " + r._3.formatted("%2.4f") + " | " + r._4.formatted("%2.4f")
      + " | " + r._5.formatted("%2.4f") + " | " + r._6.formatted("%2.4f")))
  }

  // *************************
  // * SIMILARITY MEASURES
  // *************************

  /**
    * The correlation between two vectors A, B is
    *   cov(A, B) / (stdDev(A) * stdDev(B))
    *
    * This is equivalent to
    *   [n * dotProduct(A, B) - sum(A) * sum(B)] /
    *     sqrt{ [n * norm(A)^2 - sum(A)^2] [n * norm(B)^2 - sum(B)^2] }
    */
  def correlation(size : Double, dotProduct : Double, ratingSum : Double,
                  rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double) = {

    val numerator = size * dotProduct - ratingSum * rating2Sum
    val denominator = scala.math.sqrt(size * ratingNormSq - ratingSum * ratingSum) *
      scala.math.sqrt(size * rating2NormSq - rating2Sum * rating2Sum)

    numerator / denominator
  }

  /**
    * Regularize correlation by adding virtual pseudocounts over a prior:
    *   RegularizedCorrelation = w * ActualCorrelation + (1 - w) * PriorCorrelation
    * where w = # actualPairs / (# actualPairs + # virtualPairs).
    */
  def regularizedCorrelation(size : Double, dotProduct : Double, ratingSum : Double,
                             rating2Sum : Double, ratingNormSq : Double, rating2NormSq : Double,
                             virtualCount : Double, priorCorrelation : Double) = {

    val unregularizedCorrelation = correlation(size, dotProduct, ratingSum, rating2Sum, ratingNormSq, rating2NormSq)
    val w = size / (size + virtualCount)

    w * unregularizedCorrelation + (1 - w) * priorCorrelation
  }

  /**
    * The cosine similarity between two vectors A, B is
    *   dotProduct(A, B) / (norm(A) * norm(B))
    */
  def cosineSimilarity(dotProduct : Double, ratingNorm : Double, rating2Norm : Double) = {
    dotProduct / (ratingNorm * rating2Norm)
  }

  /**
    * The Jaccard Similarity between two sets A, B is
    *   |Intersection(A, B)| / |Union(A, B)|
    */
  def jaccardSimilarity(usersInCommon : Double, totalUsers1 : Double, totalUsers2 : Double) = {
    val union = totalUsers1 + totalUsers2 - usersInCommon
    usersInCommon / union
  }

}
