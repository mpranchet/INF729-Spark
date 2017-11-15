package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer}
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.{StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}



object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._



    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegardé précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

    // lire le parquet
    val df = spark.read.parquet("./data/prepared_trainingset")

    /** TF-IDF **/

    // a) Premier stage : Tokenization

    val tokenizer = new RegexTokenizer()
        .setPattern("\\W+")
        .setGaps(true)
        .setInputCol("text")
        .setOutputCol("tokens")

    // b) 2eme stage : Remove Stop Words

    StopWordsRemover.loadDefaultStopWords("english")
    val remover = new StopWordsRemover()
        .setInputCol("tokens")
        .setOutputCol("filtered")

    // c) 3eme stage : TF

    val cvModel = new CountVectorizer()
        .setInputCol("filtered")
        .setOutputCol("rawFeatures")

    // d) 4eme stage : IDF

    val idf = new IDF()
        .setInputCol("rawFeatures")
        .setOutputCol("tfidf")


    /** Variables Catégorielles : **/


    // e) 5ème stage : convertir country2

    val indexer = new StringIndexer()
        .setInputCol("country2")
        .setOutputCol("country_indexed")
        .setHandleInvalid("skip")


    // f) 6ème stage : convertir currency2

    val indexerCurr = new StringIndexer()
        .setInputCol("currency2")
        .setOutputCol("currency_indexed")
        .setHandleInvalid("skip")


    /** VECTOR ASSEMBLER **/

    // g) 7eme stage : Vector Assembler

    val assembler = new VectorAssembler()
        .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
        .setOutputCol("features")

    /** MODEL **/

    // h) 8eme stage : model

    val lr = new LogisticRegression()
        .setElasticNetParam(0.0)
        .setFitIntercept(true).setFeaturesCol("features")
        .setLabelCol("final_status")
        .setStandardization(true)
        .setPredictionCol("predictions")
        .setRawPredictionCol("raw_predictions")
        .setThresholds(Array(0.7, 0.3))
        .setTol(1.0e-6)
        .setMaxIter(300)


    /** PIPELINE **/

    // i) Création du pipeline

    val pipeline = new Pipeline()
        .setStages(Array(tokenizer, remover, cvModel, idf, indexer, indexerCurr, assembler, lr))


    /** TRAINING AND GRID-SEARCH **/

    // j) découper le df en train/test

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)

    // k) Prepare gridsearch :

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.

    val paramGrid = new ParamGridBuilder()
        .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
        .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
        .build()

    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.

    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("final_status")
        .setPredictionCol("predictions")
        .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        // 70% of the data will be used for training and the remaining 30% for validation.
        .setTrainRatio(0.7)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)


    // l) Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    val df_WithPredictions = model.transform(test)

    // display f1 score :
    val f1 = evaluator.evaluate(df_WithPredictions)
    println(f"F1 score : ${f1*100}%.2f%%")

    // m) display
    println("Predictions : ")
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    // save the trained model :

    // now we can save the model to disk
    model.write.overwrite().save("./sample-model")

    // we can also save the pipeline to disk
    pipeline.write.overwrite.save("./sample-pipeline")

    // and load it back
    //val pipelineModel = PipelineModel.read.load("./sample-model")


  }
}
