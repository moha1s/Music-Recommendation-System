/**
  * Created by Chelsea on 12/3/16.
  */
package clebeg.spark.action


import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Caution: ALS limits every user product must get a ID, and this ID must be smaller than Integer.MAX_VALUE
  */
object MusicRecommend {
  val rootDir = "/Users/Chelsea/Downloads/Data/";//change your path
  //test in local
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkInAction").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val rawUserArtistData = sc.textFile(rootDir + "user_artist_data.txt")
    //checking if the data beyond the MAX_VALUE,
    //println(rawUserArtistData.first())
    //println(rawUserArtistData.map(_.split(' ')(0).toDouble).stats())
    //println(rawUserArtistData.map(_.split(' ')(1).toDouble).stats())
    // artist ID match to name
    val artistById = artistByIdFunc(sc)
    //check if the artist name existed
    val aliasArtist = artistsAlias(sc)

    aslModelTest(sc, aliasArtist, rawUserArtistData, artistById)
    //check  what the user 2093760 is listening right now
    val existingProducts = rawUserArtistData.map(_.split(' ')).filter {
      case Array(userId, _, _) => userId.toInt == 2093760
    }.map{
      case Array(_, artistId, _) => {
        aliasArtist.getOrElse(artistId.toInt, artistId.toInt)
      }
    }.collect().toSet

    artistById.filter {
      line => line match {
        case Some((id, name)) => existingProducts.contains(id)
        case None => false
      }

    }.collect().foreach(println)

  }

  /**
    * get the  relationship between artist name and ID
    * if some artist names are not splitted by \t, error solution to give these date up
    *
    * @param sc
    * @return
    */
  def artistByIdFunc(sc: SparkContext): RDD[Option[(Int, String)]] = {
    val rawArtistData = sc.textFile(rootDir + "artist_data.txt")
    val artistByID = rawArtistData.map {
      line =>
        //span begin the split when meeting the first unmatched case, if only a little unsuccessful,it is the quality of data
        val (id, name) = line.span(_ != '\t')
        if (name.isEmpty) {
          None
        } else {
          try {
            //
            Some((id.toInt, name.trim))
          } catch {
            case e: NumberFormatException =>
              None
          }
        }
    }
    artistByID
  }

  /**
    * through artist_alias.txt to get artists alias
    * every line slipped by \t contains a wrong ID, and a correct ID
    * some lines dont have the wrong ID, jump
    * @param sc
    * @return
    */
  def artistsAlias(sc: SparkContext) = {
    val rawArtistAlias = sc.textFile(rootDir + "artist_alias.txt")
    val artistAlias = rawArtistAlias.flatMap { line =>
      val tokens = line.split('\t')
      if (tokens(0).isEmpty) {
        None
      } else {
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collect().toMap
    artistAlias
  }

  def aslModelTest(sc: SparkContext,
                   aliasArtist: scala.collection.Map[Int, Int],
                   rawUserArtistData: RDD[String],
                   artistById: RDD[Option[(Int, String)]] ) = {
    //
    val bArtistAlias = sc.broadcast(aliasArtist)
    // Change the repeated artist ID to a same one
    val trainData = rawUserArtistData.map{
      line =>
        val Array(userId, artistId, count) = line.split(' ').map(_.toInt)
        val finalArtistID = bArtistAlias.value.getOrElse(artistId, artistId)
        Rating(userId, finalArtistID, count)
    }.cache()
    //model training
    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    //after traning , give a recommendation music list for user
    val recommendations = model.recommendProducts(2093760, 5) // recommend 5 product to user ID-2093760
    recommendations.foreach(println)
    val recommendedProductIDs = recommendations.map(_.product).toSet
    //Output recommendation artist name
    artistById.filter {
      line => line match {
        case Some((id, name)) => recommendedProductIDs.contains(id)
        case None => false
      }
    }.collect().foreach(println)
  }

}


