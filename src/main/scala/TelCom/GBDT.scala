package TelCom

import org.apache.spark.mllib.tree.impurity
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy, BoostingStrategy}
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

/**
 * Created by SHANGMAI on 2016/11/28.
 */
object GBDT {

  val conf = new SparkConf()
  conf.setAppName("GBDT")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)


  // Load and parse the data file.
  val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
  // Split the data into training and test sets (30% held out for testing)
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))

  // Train a GradientBoostedTrees model.
  // The defaultParams for Classification use LogLoss by default.
  val boostingStrategy = BoostingStrategy.defaultParams("Regression")
  boostingStrategy.setNumIterations(3) // Note: Use more iterations in practice.

  val strategy = new Strategy(Algo.Regression,impurity.Entropy,3)
 // strategy.setCategoricalFeaturesInfo()

  boostingStrategy.setTreeStrategy(strategy)

  // Empty categoricalFeaturesInfo indicates all features are continuous.
  //boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
  val model = GradientBoostedTrees.train(trainingData, boostingStrategy)


  // Evaluate model on test instances and compute test error
  val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
  println("Test Error = " + testErr)
  println("Learned classification GBT model:\n" + model.toDebugString)

  // Save and load model
  model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
  val sameModel = GradientBoostedTreesModel.load(sc,
    "target/tmp/myGradientBoostingClassificationModel")

}

