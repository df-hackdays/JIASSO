// Databricks notebook source
// MAGIC %md 
// MAGIC 
// MAGIC %md
// MAGIC <pre>
// MAGIC <b><u>Helping CLC to bridge the gaps and improve learning experience</b></u>
// MAGIC 
// MAGIC Problem Statement:
// MAGIC 
// MAGIC "How do we manage a room of learners with different experience levels and learning speeds? CLC would like to provide support for mentors and instructors to identify learners that are struggling with pace and are not comfortable publicly voicing it. Please also consider ease of use for children or those that are newer to technology."
// MAGIC 
// MAGIC Problem Scope 1: Managing learners with different experience level 
// MAGIC Proposed Solution: Predicitng adults/children that expressed intentions to pursue coding at post secondary level or work in the technical sector versus those who want to work or study coding/technology
// MAGIC 
// MAGIC Problem Scope 2: Learners who struggle with the pace or are not comfortable publicly voicing it 
// MAGIC Proposed Solution: Predicting adults/children who want to take additional coding/technical learning or coding task at work or use coding
// MAGIC 
// MAGIC Problem Scope 3: Promoting ease of use among childrens, adults and teachers 
// MAGIC Proposed Solution 3a: Predicting adults/children/teachers who are likely to recommend CLC to freind, colleage, parent, etc 
// MAGIC Proposed Solution 3b : Predicting the teachers who are likely to make donations to CLC 
// MAGIC 
// MAGIC This assignment focusses on Problem Scope 1
// MAGIC 
// MAGIC Variable Defintion 
// MAGIC 
// MAGIC Description	Code
// MAGIC Ladies Learning Code Workshop : X1
// MAGIC Kids Learning Code Workshop : X2
// MAGIC Girls Learning Code Workshop : X3
// MAGIC Teachers Learning Code Workshop : X4
// MAGIC Teens Learning Code Workshop : X5
// MAGIC Teens Learning Code: Hackathon : X6
// MAGIC Teens Learning Code: Meetup : X7
// MAGIC Code Mobile Visit: School K-8 : X8
// MAGIC Code Mobile Visit: School 9-12 : X9
// MAGIC Code Mobile Visit: Community Event : X10
// MAGIC Kids Learning Code: Summer Camp : X11
// MAGIC Kids Learning Code: March Break Camp : X12
// MAGIC Ladies Learning Code: Digital Skills for Beginners : X13
// MAGIC Teens Learning Code: Teen Ambassador : X14
// MAGIC Ladies Learning Code: National Learn to Code Day : X15
// MAGIC Girls Learning Code: Girls Learning Code Day : X16
// MAGIC Teachers Learning Code: TeacherCon: X17
// MAGIC Canada Learning Code Day: Workshop : X18
// MAGIC adult-Find new job opportunity through the CLC job board : X19
// MAGIC adult-Find new job opportunity through a new CLC connection : X20
// MAGIC adult-Encouraged others in my life to learn about coding/tech : X21
// MAGIC adult-Use coding/tech in my personal life : X22
// MAGIC adult-Engage in additional coding/tech learning:X23
// MAGIC adult-More confident in my use of coding/tech : X24
// MAGIC adult-Use coding/tech in my personal life : X25
// MAGIC adult-Updated resume/LinkedIn to include coding/tech skills : X26
// MAGIC adult-Take on new coding/tech tasks at work : X27
// MAGIC adult-Apply for a job opportunity involving coding/tech (promotion, new role, etc.) : X28
// MAGIC adult-Likely to recommend to: friend, colleague, parent/guardian or Child/youth : X29
// MAGIC kids-Wanted to know more about coding/tech : X30
// MAGIC kids-Expressed intention to pursue coding/tech at the post-secondary level : X31
// MAGIC kids-Expressed intention to work in the coding/tech sector : X32
// MAGIC kids-More confident when I code or use tech : X33
// MAGIC kids-I want to take coding/tech at college/university : X34
// MAGIC kids-I want to work in coding/tech when I grow up : X35
// MAGIC kids-Likely to recommend to guardian, friend or teacher : X36
// MAGIC teachers-Use(d) tech/coding to teach in my classroom : X37
// MAGIC teachers-Applied for a job opportunity involving coding/tech (promotion, new role, etc.) : X38
// MAGIC teachers-Updated my resume/LinkedIn to include coding/tech skills : X39
// MAGIC teachers-Use(d) coding/tech in my personal life : X40
// MAGIC teachers-Encouraged others in my life to learn about coding/tech : X41
// MAGIC teachers-Were more confident in my use of coding/tech : X42
// MAGIC teachers-Engaged in additional coding/tech learning	X43
// MAGIC teachers-Found a new job opportunity through the CLC job board : X44
// MAGIC teachers-Found a new job opportunity through a new CLC connection. : X45
// MAGIC teachers-Likely to recommend to friend, colleague, parent/guardian or Child/youth learners	: X46
// MAGIC teachers-Likely to consider making a donation to CLC : X47
// MAGIC Learner Type : Learner Type
// MAGIC Target Population	:Target Population
// MAGIC Target Variable for Recommendation :Tar_3a
// MAGIC </pre>

// COMMAND ----------

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, Row, DataFrame, Column}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.DenseVector

import org.apache.spark.ml.feature.{Imputer, ImputerModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, IndexToString} 
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}

import org.apache.spark.ml.feature.{Bucketizer,Normalizer}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification._
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel,DecisionTreeClassifier}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.NaiveBayes

//Implement PCA
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

import scala.collection.mutable
import com.microsoft.ml.spark.{LightGBMClassifier,LightGBMClassificationModel}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostEstimator, XGBoostClassificationModel}

// COMMAND ----------

//set up the syntax for spark sql
val sparkImplicits = spark
import sparkImplicits.implicits._

// COMMAND ----------

val HackDF = (sqlContext.read.option("header","true")
                             .option("inferSchema","true")
                             .format("csv")
                             .load("/FileStore/tables/Hackdata/hack_data_2.csv"))

// COMMAND ----------

HackDF.printSchema()

// COMMAND ----------

/* There are No Missing Values, So we can perform some exploratory Data analysis */

HackDF.createOrReplaceTempView("HackData_2")

// COMMAND ----------

// MAGIC %md <b> Exploratory Data Analysis </b>

// COMMAND ----------

// MAGIC %md First we explore - if the Adult would like to take the coding task at work?

// COMMAND ----------

// MAGIC %sql 
// MAGIC select tar_1, count(X27) as Interested_to_code  from HackData_2 group by tar_1 order by tar_1;

// COMMAND ----------

// MAGIC %md Next we explore - how likely this candidate will apply for a new job opportunity

// COMMAND ----------

// MAGIC %sql 
// MAGIC select tar_1, count(X28) as Is_Applicant  from HackData_2 group by tar_1 order by tar_1;

// COMMAND ----------

// MAGIC %md Feature Selection For Problem 1 (Room with Learners at different levels) : We will now filter down the relevant features

// COMMAND ----------

// MAGIC %md Split the Dataset into Training and Test Set

// COMMAND ----------

val Array(training, test) = HackDF.randomSplit(Array(0.8, 0.2),seed = 12345)

// Going to cache the data to make sure things stay snappy!
training.cache()
test.cache()

// COMMAND ----------

val assembler = new VectorAssembler().setInputCols(Array("X2","X3","X5","X6","X7","X8","X9","X10","X11","X12","X14","X16","X18","PLE","X30","X33","X34","X35","X36","Child","Youth")).setOutputCol("features")

// COMMAND ----------

val FeaturesPipeline = (new Pipeline()
  .setStages(Array(assembler)))

val trainingFit = FeaturesPipeline.fit(training)
val trainingFeatures = trainingFit.transform(training)
val testFeatures = trainingFit.transform(test)

// COMMAND ----------

// MAGIC %md Next we scale the features to normalize the data

// COMMAND ----------

// Add a standard scaler to scale the features before applying PCA
import org.apache.spark.ml.feature.StandardScaler

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false)

val scaler_fit = scaler.fit(trainingFeatures)
val scaler_training = scaler_fit.transform(trainingFeatures)
val scaler_test = scaler_fit.transform(testFeatures)

// COMMAND ----------

// MAGIC %md Model Building

// COMMAND ----------

// Create XGBoost Classifier, Random Forest Classifier, Light GBM and SVC
val rnf_model = new RandomForestClassifier().setLabelCol("Tar_1").setFeaturesCol("scaledFeatures")
val lgbm_model = new LightGBMClassifier().setLabelCol("Tar_1").setFeaturesCol("scaledFeatures")
val svc_model = new LinearSVC().setLabelCol("Tar_1").setFeaturesCol("scaledFeatures")
val log_model = new LogisticRegression().setLabelCol("Tar_1").setFeaturesCol("scaledFeatures")

// COMMAND ----------

// Setup the binary classifier evaluator
val evaluator_binary = (new BinaryClassificationEvaluator()
  .setLabelCol("Tar_1")
  .setRawPredictionCol("Prediction")
  .setMetricName("areaUnderROC"))

// COMMAND ----------

// MAGIC %md Random Forest Model

// COMMAND ----------

// Fit the model on Random Forest
val fit_rnf = rnf_model.fit(scaler_training)
val train_pred_rnf = fit_rnf.transform(scaler_training).selectExpr("Prediction", "Tar_1",
    """CASE Prediction = Tar_1
  WHEN true then 1
  ELSE 0
END as equal""")

//Random Forest - Training set
evaluator_binary.evaluate(train_pred_rnf)

// COMMAND ----------

// Print the Random Forest Model Parameters
println("Printing out the model Parameters:")
println(rnf_model.explainParams)
println("-"*20)

// COMMAND ----------

// Accuracy on Test Set
val holdout_test_rnf = fit_rnf
  .transform(scaler_test)
  .selectExpr("Prediction", "Tar_1",
    """CASE Prediction = Tar_1
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_rnf)

// COMMAND ----------

// MAGIC %md Light GBM Model

// COMMAND ----------

// Fit the model on Light GBM Forest
val fit_lgbm = lgbm_model.fit(scaler_training)
val train_pred_lgbm = fit_lgbm.transform(scaler_training).selectExpr("Prediction", "Tar_1",
    """CASE Prediction = Tar_1
  WHEN true then 1
  ELSE 0
END as equal""")

// Light GBM - Training set
evaluator_binary.evaluate(train_pred_lgbm)

// COMMAND ----------

// Print the Light GBM Model Parameters
println("Printing out the model Parameters:")
println(lgbm_model.explainParams)
println("-"*20)

// COMMAND ----------

// Now check the accuracy on test set
val holdout_lgbm = fit_lgbm
  .transform(scaler_test)
  .selectExpr("Prediction", "Tar_1",
    """CASE Prediction = Tar_1
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_lgbm)

// COMMAND ----------

// MAGIC %md Logistic Regression Model

// COMMAND ----------

// Fit the model on logistic regression model
val fit_log = log_model.fit(scaler_training)
val train_pred_log = fit_log.transform(scaler_training).selectExpr("Prediction", "cast(Tar_1 as Double) Tar_1",
    """CASE Prediction = Tar_1
  WHEN true then 1
  ELSE 0
END as equal""")

//logistic regression Model - Training set
evaluator_binary.evaluate(train_pred_log)

// COMMAND ----------

// Print the Logistic regression model Parameters
println("Printing out the model Parameters:")
println(log_model.explainParams)
println("-"*20)

// COMMAND ----------

// Accuracy on Test Set
val holdout_test_log = fit_log
  .transform(scaler_test)
  .selectExpr("Prediction", "cast(Tar_1 as Double) Tar_1",
    """CASE Prediction = X47
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_log)

// COMMAND ----------

/* Get the Variable importance for Random Forest */
val importances = fit_rnf
  .asInstanceOf[RandomForestClassificationModel]
  .featureImportances

// COMMAND ----------

// MAGIC %md
// MAGIC Some of the important variables based on importance are as follows

// COMMAND ----------

// MAGIC %md
// MAGIC <pre>
// MAGIC kids-More confident when I code or use tech           0.887724391 
// MAGIC kids-I want to take coding/tech at college/university 0.012220235
// MAGIC kids-Wanted to know more about coding/tech            0.0103764
// MAGIC Prior Learning Experience                             0.010291065
// MAGIC </pre>
