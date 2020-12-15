name := "BigData and Machine Learning"
 
version := "1.0"

scalaVersion := "2.12.12"

// javaOptions += "-Xmx24G"

run / fork := true
run / connectInput := true

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.0"
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "3.0.0"
// libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0" % "provided"
// libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0" % "provided"
// libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.0" % "provided"
// libraryDependencies += "org.apache.spark" %% "spark-graphx" % "3.0.0" % "provided"
