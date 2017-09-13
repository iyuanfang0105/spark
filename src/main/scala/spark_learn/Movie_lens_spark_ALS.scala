package spark_learn

import scala.util.Random
import scala.math
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._
import org.apache.spark.sql.SQLContext

import scala.collection.mutable.ArrayBuffer

/**
  * Created by toddmcgrath on 6/15/16.
  */
object Movie_lens_spark_ALS {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    val sparkConf = new SparkConf()
    sparkConf.setAppName("recommend film sys")
    sparkConf.setMaster("local[2]")

    val sparkContext = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sparkContext)


    // load the movies info, format: (movieId, movieName)
    val movie_lens_data_path = "/home/meizu/WORK/public_dataset/movie_lens/ml-10M100K"
    val movies = sparkContext.textFile(movie_lens_data_path + "/movies.dat").map { line =>
      val fields = line.split("::")
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect.toMap
    // println("\n\n**************** Movies: *************** \n\n")
    // movies.take(10).foreach(println)

    // load rating info, format: (timestamp % 10, Rating(userId, movieId, rating))
    val ratings: RDD[(Long, Rating)] = sparkContext.textFile(movie_lens_data_path + "/ratings.dat").map { line =>
      val fields = line.split("::")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }
    // println("\n\n**************** Ratings 10 samples: *************** \n\n")
    // ratings.take(10).foreach(println)
    val rating_num = ratings.count()
    val user_num = ratings.map(_._2.user).distinct.count
    val movie_num = ratings.map(_._2.product).distinct.count
    println("\n====>>>>  Ratings Counts: " + rating_num)
    println("====>>>>  User Counts: " + user_num)
    println("====>>>>  Movie Counts: " + movie_num)

    // get the most rated movies and let user to rate
    val most_rated_movie_Ids = ratings.map(_._2.product) // extract movie ids
                                .countByValue      // count ratings per movie
                                .toSeq             // convert map to Seq
                                .sortBy(- _._2)    // sort by rating count
                                .take(50)          // take 50 most rated
                                .map(_._1)         // get their ids

    val selected_movies = Random.shuffle(most_rated_movie_Ids).take(10).map(x => (x, movies(x))).toSeq
    // selected_movies.foreach(println)
    // val random = new Random(0)
    // val selected_movies = most_rated_movie_Ids.filter(x => random.nextDouble() < 0.2)
    //                          .map(x => (x, movies(x)))
    //                          .toSeq

    val my_ratings = elicitate_ratings(selected_movies)
    val my_ratings_rdd: RDD[Rating] = sparkContext.parallelize(my_ratings.toSeq)
    //val my_ratings_rdd = sparkContext.parallelize(Seq(my_ratings))

    // split the data to train, valid, test
    val num_partitions = 2
    val train_set = ratings.filter(v => v._1 < 6).values.union(my_ratings_rdd).repartition(num_partitions)
    val valid_set = ratings.filter(v => v._1 >= 6 && v._1 <8).values.repartition(num_partitions)
    val test_set = ratings.filter(v => v._1 >= 8).values

    val train_num = train_set.count()
    val valid_num = valid_set.count()
    val test_num = test_set.count()

    println("\n====>>>>Training: " + train_num + ", validation: " + valid_num + ", test: " + test_num)

    // train models and evaluate them on the validation set

    val ranks = List(8, 12)
    val lambdas = List(0.1, 10.0)
    val iters_num = List(10, 20)
    var best_model: Option[MatrixFactorizationModel] = None
    var best_validation_rmse = Double.MaxValue
    var best_rank = 0
    var best_lambda = -1.0
    var best_numIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- iters_num) {
      val model = ALS.train(train_set, rank, numIter, lambda)
      // val validation_RMSE = compute_RMSE_user(model, valid_set, valid_num)
      val validation_RMSE = compute_RMSE_spark(model, valid_set).rootMeanSquaredError
      println("\n====>>>> RMSE (validation) = " + validation_RMSE + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validation_RMSE < best_validation_rmse) {
        best_model = Some(model)
        best_validation_rmse = validation_RMSE
        best_rank = rank
        best_lambda = lambda
        best_numIter = numIter
      }
    }

    // test the best model performance
    val test_RMSE = compute_RMSE_spark(best_model.get, test_set)
    println("\n====>>>> The best model was trained with rank = " + best_rank + " and lambda = " + best_lambda
      + ", and numIter = " + best_numIter + ", and its RMSE on the test set is " + test_RMSE + ".")

    // make recommendation to me
    val my_rated_movie_ids = my_ratings.map(_.product).toSet
    val candidates = sparkContext.parallelize(movies.keys.filter(!my_rated_movie_ids.contains(_)).toSeq)
    val recommendations = best_model.get
      .predict(candidates.map((0, _)))
      .collect
      .sortBy(- _.rating)
      .take(10)

    var i = 1
    println("Movies recommended for you:")
    recommendations.foreach { r =>
      println("\n====>>>> %2d".format(i) + ": " + movies(r.product))
      i += 1
    }

  }

  /**
    * Compute RMSE (Root Mean Squared Error)
    * spark default implementation
    */
  def compute_RMSE_spark(model: MatrixFactorizationModel, data: RDD[Rating]) : RegressionMetrics = {
    val validation_result: RDD[((Int, Int), Double)] = model.predict(data.map(x => (x.user, x.product))).map(v => ((v.user, v.product), v.rating))
    // userid, product, gt_rating, pred_rating
    val validation_GT_PRED: RDD[((Int, Int), (Double, Double))] = data.map(v => ((v.user, v.product), v.rating)).join(validation_result)
    val validation_GT_PRED_rating: RDD[(Double, Double)] = validation_GT_PRED.values
    val validation_metrics = new RegressionMetrics(validation_GT_PRED_rating)
    return validation_metrics
  }

  /**
    * Compute RMSE (Root Mean Squared Error).
    * user implementation
    */
  def compute_RMSE_user(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictions_and_ratings = predictions.map(x => ((x.user, x.product), x.rating)).join(data.map(x => ((x.user, x.product), x.rating))).values

    val RMSE: Double = math.sqrt(predictions_and_ratings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
    return RMSE
  }

  /** Elicitate ratings from command-line. */
  def elicitate_ratings(movies: Seq[(Int, String)]): ArrayBuffer[Rating] = {
    val prompt = "Please rate the following movie (1-5 (best), or 0 if not seen):"
    println(prompt)
    var rating: ArrayBuffer[Rating] = new ArrayBuffer[Rating]()
    for(item: (Int, String) <- movies){
      print(item._2 + ": ")
      try {
        val r = Console.readInt
        if (r < 0 || r > 5) {
          println(prompt)
        } else {
          rating.append(Rating(0, item._1, r))
        }
      } catch {
        case e: Exception => println(prompt)
      }
    }
    return rating
  }
}
