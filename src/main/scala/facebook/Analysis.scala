package facebook

import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.Row
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import swiftvis2.plotting._
import org.apache.spark.sql.Dataset
import swiftvis2.plotting.renderer.SwingRenderer
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.Pipeline
import swiftvis2.plotting.styles.HistogramStyle
import org.apache.spark.sql.Encoders._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import spire.std.seq
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.catalyst.expressions.aggregate.Corr
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.graphx.GraphLoader

case class MaskData(fp: String, never: Double, rarely: Double, sometimes: Double, freq: Double, always: Double)
case class SRDState(srd: String, state: String)
case class SCILocs(fips1: String, fips2: String, sci: Int, lat1: Double, lon1: Double, lat2: Double, lon2: Double)
case class SCIDist(fips1: String, fips2: String, sci: Int, dist: Double)
case class SCIDist2(fips1: String, fips2: String, sci: Double, dist: Double)
case class SeriesData(sid: String, areaType: String, areaCode: String, measureCode: String, srdCode: String, title: String)
case class BLSData(sid: String, year: Int, period: String, value: Double, footnotes: String)
case class SCI(fips1: String, fips2: String, sci: Int)
case class StToState(fullState: String, st: String)
case class AdjCounties(locName: String, locFips: String, neighborsFips: List[String])
case class InvSci(fips1: String, fips2: String, invSci: Double)
case class ElectionData(lineNum: Int, dVotes: Int, rVotes: Int, totalVotes: Int, perDem: Double, perRep: Double, diff: Int, perPtDiff: Double, state: String, county: String, fipsCode: String)
case class SCIDoubleVal(fips1: String, fips2: String, sci: Int, val1: Double, val2: Double)
case class SCIDiff(fips1: String, fips2: String, sci: Int, diff: Double)
case class FullData(fips1: String, fips2: String, sci: Double, cov: Double, elec: Double, dist: Double, ur: Double)

object Analysis {
  def main(args: Array[String]): Unit = {

        val spark = SparkSession.builder().master("local[*]").appName("Final Project").getOrCreate()
        import spark.implicits._
    
        spark.sparkContext.setLogLevel("WARN")

        val stToState = spark.read.schema(Encoders.product[StToState].schema).option("header", true).option("delimiter", "\t").csv("data/stateST.txt").as[StToState].cache()
        val srdToStates = spark.read.schema(Encoders.product[SRDState].schema).option("header", true).option("delimiter", "\t").csv("data/la.state_region_division.txt").as[SRDState].cache()
        val fipsToCounties = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").csv("data/county_centers.csv").cache()
        val maskData = spark.read.schema(Encoders.product[MaskData].schema).option("header", true).option("delimiter", ",").csv("data/masks.txt").as[MaskData].cache()
        val sciData = spark.read.schema(Encoders.product[SCI].schema).option("header", true).option("delimiter", "\t").csv("data/county_county_aug2020.tsv").as[SCI].filter('fips1 !== 'fips2).cache()
        val src = scala.io.Source.fromFile("data/county_adjacency.txt")
        val neighborCounties = src.getLines.mkString("\n").split("\n"+""""""").map{ section =>
            val fixed = if (!section.startsWith(""""""")) """""""+section else section
            val lines = fixed.split("\n")
            val curr = lines.head.split('\t')
            val currName = curr(0).substring(1,curr(0).length()-1)
            val currFips = curr(curr.length-1)
            val rest = (List(curr(1))++lines.tail).map(line => line.split('\t').last).toList.filter(x => currFips != x)
            AdjCounties(currName, currFips, rest)
        }.toSeq.toDF().as[AdjCounties].cache()
        src.close()
        
        val sKey = spark.read.textFile("data/la.series").map { line =>
          val p = line.split("\t").map(_.trim)
          SeriesData(p(0), p(1), p(2), p(3), p(5), p(6))
        }.cache()

        val dataArk = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.10.Arkansas").as[BLSData].cache()
        val dataCali = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.11.California").as[BLSData].cache()
        val dataColo = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.12.Colorado").as[BLSData].cache()
        val dataConn = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.13.Connecticut").as[BLSData].cache()
        val dataDela = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.14.Delaware").as[BLSData].cache()
        val dataDC = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.15.DC").as[BLSData].cache()
        val dataFlor = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.16.Florida").as[BLSData].cache()
        val dataGA = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.17.Georgia").as[BLSData].cache()
        val dataID = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.19.Idaho").as[BLSData].cache()
        val dataIll = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.20.Illinois").as[BLSData].cache()
        val dataInd = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.21.Indiana").as[BLSData].cache()
        val dataIowa = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.22.Iowa").as[BLSData].cache()
        val dataKan = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.23.Kansas").as[BLSData].cache()
        val dataKent = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.24.Kentucky").as[BLSData].cache()
        val dataLouis = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.25.Louisiana").as[BLSData].cache()
        val dataMaine = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.26.Maine").as[BLSData].cache()
        val dataMary = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.27.Maryland").as[BLSData].cache()
        val dataMass = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.28.Massachusetts").as[BLSData].cache()
        val dataMich = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.29.Michigan").as[BLSData].cache()
        val dataMN = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.30.Minnesota").as[BLSData].cache()
        val dataMiss = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.31.Mississippi").as[BLSData].cache()
        val dataMiso = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.32.Missouri").as[BLSData].cache()
        val dataMon = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.33.Montana").as[BLSData].cache()
        val dataNeb = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.34.Nebraska").as[BLSData].cache()
        val dataNev = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.35.Nevada").as[BLSData].cache()
        val dataNH = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.36.NewHampshire").as[BLSData].cache()
        val dataNJ = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.37.NewJersey").as[BLSData].cache()
        val dataNY = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.39.NewYork").as[BLSData].cache()
        val dataNCar = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.40.NorthCarolina").as[BLSData].cache()
        val dataNDak = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.41.NorthDakota").as[BLSData].cache()
        val dataOH = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.42.Ohio").as[BLSData].cache()
        val dataOK = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.43.Oklahoma").as[BLSData].cache()
        val dataOreg = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.44.Oregon").as[BLSData].cache()
        val dataPenn = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.45.Pennsylvania").as[BLSData].cache()
        val dataPR = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.46.PuertoRico").as[BLSData].cache()
        val dataRho = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.47.RhodeIsland").as[BLSData].cache()
        val dataSCar = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.48.SouthCarolina").as[BLSData].cache()
        val dataSDak = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.49.SouthDakota").as[BLSData].cache()
        val dataTenn = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.50.Tennessee").as[BLSData].cache()
        val dataUtah = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.52.Utah").as[BLSData].cache()
        val dataVer = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.53.Vermont").as[BLSData].cache()
        val dataVirg = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.54.Virginia").as[BLSData].cache()
        val dataWash = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.56.Washington").as[BLSData].cache()
        val dataWV = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.57.WestVirginia").as[BLSData].cache()
        val dataWisc = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.58.Wisconsin").as[BLSData].cache()
        val dataWy = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.59.Wyoming").as[BLSData].cache()
        val dataAl = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.7.Alabama").as[BLSData].cache()
        val dataAri = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.9.Arizona").as[BLSData].cache()
        val dataTX = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.51.Texas").as[BLSData].cache()
        val dataNM = spark.read.schema(Encoders.product[BLSData].schema).option("header",true).option("delimiter", "\t").csv("data/la.data.38.NewMexico").as[BLSData].cache()
        
        val fullBLSData = Seq(dataAri,dataAl,dataWy,dataWisc,dataWV,dataWash,dataVirg,dataVer,dataUtah,dataTenn,dataSDak,dataSCar,dataRho,dataPR,dataPenn,dataOreg,dataOK,dataOH,dataMon,dataMiso,dataMiss,dataMN,dataMich,dataMass,dataMary,dataMaine,dataNDak,dataNCar,dataNY,dataNJ,dataNH,dataNev,dataNeb,dataLouis,dataKent,dataKan,dataIowa,dataInd,dataIll,dataID,dataGA,dataFlor,dataDC,dataDela,dataConn,dataColo,dataCali,dataArk,dataNM, dataTX).reduce(_ unionAll _).select(trim('sid) as "sid", 'year, 'period, 'value, 'footnotes).as[BLSData].cache()

        val elections = spark.read.textFile("data/2016_US_County_Level_Presidential_Results.csv").filter(x => !x.startsWith(",")).map { line =>
          val p = line.split(",")
          if(p.length == 12){
            ElectionData(p(0).toInt, p(1).toDouble.toInt, p(2).toDouble.toInt, p(3).toDouble.toInt, p(4).toDouble, p(5).toDouble, p(6).substring(1).toInt*1000+p(7).substring(0,p(7).length()-1).toInt, p(8).substring(0,p(8).length()-1).toDouble, p(9), p(10), p(11))
          } else if(p.length == 13) {
            ElectionData(p(0).toInt, p(1).toDouble.toInt, p(2).toDouble.toInt, p(3).toDouble.toInt, p(4).toDouble, p(5).toDouble, p(6).substring(1).toInt*1000000+p(7).toInt*1000+p(8).substring(0,p(8).length()-1).toInt, p(9).substring(0,p(9).length-1).toDouble, p(10), p(11), p(12))
          } else {
            ElectionData(p(0).toInt, p(1).toDouble.toInt, p(2).toDouble.toInt, p(3).toDouble.toInt, p(4).toDouble, p(5).toDouble, p(6).toInt, p(7).substring(0,p(7).length() - 1).toDouble, p(8), p(9), p(10))
          }
        }.filter('state =!= "AK" && 'state =!= "HI").cache()

        // val lfs = fullBLSData.filter('year === 2010 && 'period === "M13" && 'sid.endsWith("06")).as[BLSData]
        // val countyLfs = lfs.join(sKey.filter('areaType === "F"), lfs("sid") === sKey("sid")).select("value", "title", "srdCode")
        // val countyStateLfs = countyLfs.join(srdToStates, srdToStates("srd") === countyLfs("srdCode")).select("value", "title", "state")
        // countyStateLfs.show()
        // val electionStates = elections.join(stToState, stToState("st") === elections("state")).select("county", "fullState", "fipsCode")
        // electionStates.show()
        // val lfsFips = electionStates.join(countyStateLfs, countyStateLfs("title").contains(electionStates("county")) && countyStateLfs("state") === electionStates("fullState")).select("value", "fipsCode")
        // val sciPop = sciData.join(lfsFips, lfsFips("fipsCode") === sciData("fips1")).select("fips1", "fips2", "sci", "value")
        // sciPop.show()
        // println(sciPop.count())






        // val favorites = sciData.groupBy('fips1).max("sci")
        // val favoritesWithCoords = favorites.join(fipsToCounties, favorites("fips1") === fipsToCounties("fips")).select("clon10", "clat10", "max(sci)").filter('clon10 < -60 && 'clon10 > -130 && 'clat10 > 20 && 'clat10 < 50).collect()

        // val bins = (1.0 to 70000000.0 by 1000000).toArray
        // val hist = favorites.select("max(sci)").as[Double].rdd.histogram(bins,false)
        // val histPlot = Plot.histogramPlot(bins, hist, RedARGB, false, "Occurrences of Maximum SCIs", "SCI", "Count")
        // SwingRenderer(histPlot, 1000, 1000, true)

        // val bins = (1000.0 to 1000000.0 by 1000).toArray
        // val hist = sciData.select("sci").as[Double].rdd.histogram(bins,false)
        // val histPlot = Plot.histogramPlot(bins, hist, RedARGB, false, "Occurrences of SCIs", "SCI", "Count")
        // SwingRenderer(histPlot, 1000, 1000, true)

        // val gradFav = ColorGradient(50000.0 -> BlackARGB, 1000000.0 -> BlueARGB, 20000000.0 -> GreenARGB, 30000000.0 -> RedARGB)
        
        // val scisFav = favoritesWithCoords.map(_.getInt(2).toDouble)
        // val longsFav = favoritesWithCoords.map(_.getString(0).toDouble)
        // val latsFav = favoritesWithCoords.map(_.getString(1).toDouble)
        // val plotFav = Plot.scatterPlot(longsFav, latsFav, "Whatever", "Longitude", "Latitude", 4, scisFav.map(gradFav))
        // SwingRenderer(plotFav, 1000, 780, true)





          def oneCounty(fips: String, name: String): Unit = {
            val from20183 = sciData.filter('fips1 === fips)
            println(from20183.count())
            val coords20183 = from20183.join(fipsToCounties, from20183("fips2") === fipsToCounties("fips")).select("clon10", "clat10", "sci").filter('clon10 < -60 && 'clon10 > -130 && 'clat10 > 20 && 'clat10 < 50).collect()
            val grad20183 = ColorGradient(1.0 -> YellowARGB, 10000.0 -> GreenARGB, 200000.0 -> BlueARGB, 5000000.0 -> RedARGB)
            val scis20183 = coords20183.map(_.getInt(2).toDouble)
            val longs20183 = coords20183.map(_.getString(0).toDouble)
            val lats20183 = coords20183.map(_.getString(1).toDouble)
            val plot20183 = Plot.scatterPlot(longs20183, lats20183, "SCIs of " + name + " County, Minnesota", "Longitude", "Latitude", 4, scis20183.map(grad20183))
            SwingRenderer(plot20183, 1000, 780, true)
          }






        def haversine(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double = {
          val latRad1 = lat1/(180/math.Pi)
          val latRad2 = lat2/(180/math.Pi)
          val lonRad1 = lon1/(180/math.Pi)
          val lonRad2 = lon2/(180/math.Pi)
          math.acos(math.sin(latRad1)*math.sin(latRad2)+math.cos(latRad1)*math.cos(latRad2)*math.cos(lonRad2-lonRad1))*6378.8
        }

        // val loc1SciData = sciData.join(fipsToCounties, fipsToCounties("fips") === sciData("fips1")).select('fips1, 'fips2, 'sci, 'pclat10 as "lat1", 'pclon10 as "lon1")
        // val locSciData = loc1SciData.join(fipsToCounties, fipsToCounties("fips") === sciData("fips2")).select('fips1, 'fips2, 'sci, 'lat1, 'lon1, 'pclat10 as "lat2", 'pclon10 as "lon2").map(r => SCILocs(r.getString(0), r.getString(1), r.getInt(2), r.getString(3).toDouble, r.getString(4).toDouble, r.getString(5).toDouble, r.getString(6).toDouble))
        // val distData = locSciData.map( x => SCIDist(x.fips1, x.fips2, x.sci, haversine(x.lat1, x.lon1, x.lat2, x.lon2))).collect()

        // val scis = distData.map(x => math.log(x.sci))
        // val dists = distData.map(x => math.log(x.dist))
        // // val latsFav = favoritesWithCoords.map(_.getString(1).toDouble)
        // val plotFav = Plot.scatterPlot(dists, scis, "Whatever", "sci", "dist", 1, BlackARGB)
        // SwingRenderer(plotFav, 1000, 1000, true)





        
        
        // val sciMask1 = sciData.join(maskData, maskData("fp") === sciData("fips1")).select('fips1, 'fips2, 'sci, 'never + 'rarely as "no1")
        // val sciMask = sciMask1.join(maskData, maskData("fp") === sciMask1("fips2")).select('fips1, 'fips2, 'sci, 'no1, 'never + 'rarely as "no2")
        // val sciMaskDiff = sciMask.map(x => SCIDiff(x.getString(0), x.getString(1), x.getInt(2), math.abs(x.getDouble(3)-x.getDouble(4)))).cache()
        // // sciMask.show()

        // val sciElec1 = sciData.join(elections, elections("fipsCode") === sciData("fips1")).select('fips1, 'fips2, 'sci, 'perDem as "perDem1")
        // val sciElec = sciElec1.join(elections, elections("fipsCode") === sciElec1("fips2")).select('fips1, 'fips2, 'sci, 'perDem1, 'perDem as "perDem2")
        // val sciElecDiff = sciElec.map(x => SCIDiff(x.getString(0), x.getString(1), x.getInt(2), math.abs(x.getDouble(3)-x.getDouble(4)))).cache()
        // // sciElec.show()

        // val ur2010 = fullBLSData.filter('year === 2010 && 'period === "M13" && 'sid.endsWith("03")).as[BLSData]
        // val countyURs = ur2010.join(sKey.filter('areaType === "F"), ur2010("sid") === sKey("sid")).select("value", "title", "srdCode")
        // val countyStateUR = countyURs.join(srdToStates, srdToStates("srd") === countyURs("srdCode")).select("value", "title", "state")
        // val countyToFips = elections.join(stToState, stToState("st") === elections("state")).select("county", "fullState", "fipsCode")
        // val urFips = countyToFips.join(countyStateUR, countyStateUR("title").contains(countyToFips("county")) && countyStateUR("state") === countyToFips("fullState")).select("value", "fipsCode")
        // val sciUR1 = sciData.join(urFips, urFips("fipsCode") === sciData("fips1")).select('fips1, 'fips2, 'sci, 'value as "ur1")
        // val sciUR = sciUR1.join(urFips, urFips("fipsCode") === sciUR1("fips2")).select('fips1, 'fips2, 'sci, 'ur1, 'value as "ur2")
        // val sciURDiff = sciUR.map(x => SCIDiff(x.getString(0), x.getString(1), x.getInt(2), math.abs(x.getDouble(3)-x.getDouble(4)))).cache()
        // // sciUR.show()

        // val sciDists = locSciData.map(x => SCIDiff(x.fips1, x.fips2, x.sci, haversine(x.lat1, x.lon1, x.lat2, x.lon2)))//.filter(x => x.fips1.substring(0,2) == x.fips2.substring(0,2))

        // val finalRegData1 = sciDists.join(sciURDiff, sciDists("fips1") === sciURDiff("fips1") && sciDists("fips2") === sciURDiff("fips2")).select(sciDists("fips1"), sciDists("fips2"), sciDists("sci"), sciDists("diff") as "dist", sciURDiff("diff") as "ur")
        // val finalRegData2 = finalRegData1.join(sciElecDiff, sciElecDiff("fips1") === finalRegData1("fips1") && sciElecDiff("fips2") === finalRegData1("fips2")).select(finalRegData1("fips1"), finalRegData1("fips2"), finalRegData1("sci"), 'dist, 'ur, 'diff as "elec")
        // val finalRegData = finalRegData2.join(sciMaskDiff, sciMaskDiff("fips1") === finalRegData2("fips1") && sciMaskDiff("fips2") === finalRegData2("fips2")).select(finalRegData2("fips1"), finalRegData2("fips2"), finalRegData2("sci"), 'dist, 'ur, 'elec, 'diff as "cov").map(x => FullData(x.getString(0), x.getString(1), math.log(x.getInt(2)), x.getDouble(6), x.getDouble(5), math.log(x.getDouble(3)), x.getDouble(4)))

        // val Array(training, test) = finalRegData.randomSplit(Array(.8,.2))
        // val testLinearRegData = new VectorAssembler().setInputCols(Array("cov")).setOutputCol("features").transform(test).cache()
        // val linearRegData = new VectorAssembler().setInputCols(Array("cov")).setOutputCol("features").transform(training).cache()
        // val linearRegFull = new VectorAssembler().setInputCols(Array("cov")).setOutputCol("features").transform(finalRegData).cache()
        // val linearReg = new LinearRegression().setFeaturesCol("features").setLabelCol("sci").setMaxIter(10).setPredictionCol("SciPred")
        // val linearRegModel = linearReg.fit(linearRegData)
        // println(linearRegModel.coefficients+" "+linearRegModel.intercept)
        // val withLinearFit = linearRegModel.transform(testLinearRegData)
        // val fittedData = linearRegModel.transform(linearRegFull)
        // fittedData.show()
        // println(println(s"RMSE: ${linearRegModel.summary.rootMeanSquaredError}"))

        // val inputs = fittedData.select('cov).map(d => d.getDouble(0)*linearRegModel.coefficients(0)+linearRegModel.intercept).collect()
        // val preds = fittedData.select("SciPred").as[Double].collect()
        // val scis = fittedData.select('sci).as[Double].collect()
        // val plot = Plot.scatterPlots(Seq((inputs, scis, BlackARGB, 1), (inputs, preds, RedARGB, 2)),"Regression 2.5", "adjusted f(masks)", "log(SCI) and Prediction")
        // SwingRenderer(plot, 1000, 1000, true)





        // val invSciData = sciData.filter(x => !x.fips1.startsWith("02") && !x.fips2.startsWith("02") && !x.fips2.startsWith("15") && !x.fips1.startsWith("15")).map(x => InvSci(x.fips1, x.fips2, math.pow(x.sci, -1)))
        // println(invSciData.filter(_.invSci >= 0.999).count())
        // println(invSciData.count())
        // val assembler = new VectorAssembler().setInputCols(Array("invSci")).setOutputCol("features")
        // val transformed = assembler.transform(invSciData)
        // transformed.show()
        // val bkm = new BisectingKMeans().setK(48).setSeed(1L).setFeaturesCol("features")
        // val model = bkm.fit(transformed)
        // val clustered = model.transform(transformed)
        // clustered.show()

        // val clusteredWithCoords1 = clustered.join(fipsToCounties, fipsToCounties("fips") === clustered("fips1")).select('fips1, 'fips2, 'prediction, 'clat10 as "lat1", 'clon10 as "lat2")
        // val clusteredWithCoords = clusteredWithCoords1.join(fipsToCounties, fipsToCounties("fips") === clusteredWithCoords1("fips2")).select('fips1, 'fips2, 'prediction, 'lat1, 'lon1, 'clat10 as "lat2", 'clon10 as "lon2")
        // clusteredWithCoords.show()

        val avgs = sciData.groupBy('fips1).avg("sci")
        val avgsWithCoords = avgs.join(fipsToCounties, avgs("fips1") === fipsToCounties("fips")).select("fips1", "clon10", "clat10", "avg(sci)").filter('clon10 < -60 && 'clon10 > -130 && 'clat10 > 20 && 'clat10 < 50)
        
        val lf2010 = fullBLSData.filter('year === 2010 && 'period === "M13" && 'sid.endsWith("06")).as[BLSData]
        val countyLFs = lf2010.join(sKey.filter('areaType === "F"), lf2010("sid") === sKey("sid")).select("value", "title", "srdCode")
        val countyStateLF = countyLFs.join(srdToStates, srdToStates("srd") === countyLFs("srdCode")).select("value", "title", "state")
        val countyToFips = elections.join(stToState, stToState("st") === elections("state")).select("county", "fullState", "fipsCode")
        val lfFips = countyToFips.join(countyStateLF, countyStateLF("title").contains(countyToFips("county")) && countyStateLF("state") === countyToFips("fullState")).select("value", "fipsCode")
        val avgCoordPop = avgsWithCoords.join(lfFips, lfFips("fipsCode") === avgsWithCoords("fips1")).select("fips1", "clon10", "clat10", "avg(sci)", "value")
        
        val avgCoordPopCov = maskData.join(avgCoordPop, avgCoordPop("fips1") === maskData("fp")).select("fips1", "clon10", "clat10", "avg(sci)", "value", "never", "rarely", "sometimes", "freq", "always").withColumnRenamed("avg(sci)", "avg").withColumnRenamed("clat10", "lat").withColumnRenamed("clon10", "lon").withColumnRenamed("value","pop").withColumn("lat", col("lat").cast(DoubleType)).withColumn("lon", col("lon").cast(DoubleType))
        val avgCoordPopCovElec = elections.join(avgCoordPopCov, avgCoordPopCov("fips1") === elections("fipsCode")).select("fips1", "lon", "lat", "avg", "pop", "perDem", "never", "rarely", "sometimes", "freq", "always")
        
        // val maskClassification = avgCoordPopCovElec.withColumn("maskClass", expr("CASE WHEN never + rarely + sometimes > 0.5 THEN 0.0 ELSE 1.0 END")).drop("never", "rarely", "sometimes", "freq", "always")
        // println(maskClassification.filter('maskClass === 0.0).count())
        // val sciClassification = avgCoordPopCovElec.withColumn("sciClass", expr("CASE WHEN avg < 15000 THEN 0.0 ELSE 1.0 END")).drop("avg")
        // sciClassification.show()

        // val va = new VectorAssembler().setInputCols(Array("avg", "pop", "perDem")).setOutputCol("features")
        // val transformed = va.transform(maskClassification)
        // transformed.show()
        
        // val Array(training, test) = transformed.randomSplit(Array(0.7, 0.3), seed = 1234L)
        // val model = new NaiveBayes().setLabelCol("maskClass").setFeaturesCol("features").setPredictionCol("prediction").fit(training)

        // val preds = model.transform(test)
        // preds.show()

        // val evaluator = new MulticlassClassificationEvaluator().setLabelCol("maskClass").setPredictionCol("prediction").setMetricName("accuracy")
        // val accuracy = evaluator.evaluate(preds)
        // println("Mask Classification Accuracy: " + accuracy)





        // val va = new VectorAssembler().setInputCols(Array("pop","never", "rarely", "sometimes", "freq", "always", "perDem")).setOutputCol("features")
        // val transformed = va.transform(sciClassification)
        // transformed.show()
        
        // val Array(training, test) = transformed.randomSplit(Array(0.7, 0.3), seed = 1234L)
        // val model = new NaiveBayes().setLabelCol("sciClass").setFeaturesCol("features").setPredictionCol("prediction").fit(training)

        // val preds = model.transform(test)
        // preds.show()

        // val evaluator = new MulticlassClassificationEvaluator().setLabelCol("sciClass").setPredictionCol("prediction").setMetricName("accuracy")
        // val accuracy = evaluator.evaluate(preds)
        // println("SCI Classification Accuracy: " + accuracy)




        
        val gradAvg = ColorGradient(0.0 -> BlackARGB, 1000.0 -> BlueARGB, 10000.0 -> GreenARGB, 100000.0 -> RedARGB)
        val scisFav = avgsWithCoords.map(_.getDouble(3)).collect()
        val longsFav = avgsWithCoords.map(_.getString(1).toDouble).collect()
        val latsFav = avgsWithCoords.map(_.getString(2).toDouble).collect()
        val plotFav = Plot.scatterPlot(longsFav, latsFav, "Average SCI", "Longitude", "Latitude", 4, scisFav.map(gradAvg))
        SwingRenderer(plotFav, 1000, 780, true)



        spark.close()
  }
}
