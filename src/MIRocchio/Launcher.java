/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package MIRocchio;

import Utils.EvaluationPlus;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SelectedTag;
import weka.classifiers.mi.CitationKNN;
import weka.classifiers.mi.MITI;
import weka.classifiers.mi.MILR;
import weka.classifiers.mi.MISVM;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.filters.unsupervised.attribute.MILESFilter;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;

/**
 *
 * @author Danel
 */
public class Launcher {

  /**
   * Threshold Model
   */
  final static private int thresholdModel = MIRocchio.tm_Linear;

  /**
   * Configuración del problema WIR original (hold-out)
   */
  //final static private String genericPath = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/WebMIL/less-word/tfc-tfc/";
  //final static private String fileName = "Mismo_MI-tfc-tfc-V%d-words5-40.%s.arff"; // %d = 1..9, %s = test, train
  /**
   * Configuración del problema WIR preparado para 5x5CV
   */
  //final static private String genericPath = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/IMBALANCED/5CV/wir-v%d/";
  //final static private String trainFileName = "Mismo_wir-v%d-r%d-f%d-train.arff"; // %d = 1..3, 1..5, 1..5
  //final static private String testFileName = "wir-v%d-r%d-f%d-test.arff"; // %d = 1..3, 1..5, 1..5

  /**
   * Configuración del problema WIR preparado para 10CV
   */
  final static private String genericPath = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/TREC9 (OHSUMED)/1/AttrsFixed/";
  final static private String trainFileName = "TREC9-1-f%d-train.arff"; // %d = 1..3, 1..5, 1..5
  final static private String testFileName = "TREC9-1-f%d-test.arff"; // %d = 1..3, 1..5, 1..5
  final static private int positiveLabel = 0;

  public static void testClassifier_10CV(Classifier c) {
    String specificPath = genericPath;
    EvaluationPlus eval = null;
    for (int fold = 1; fold <= 10; fold++)  {
      try {
            String trainName = String.format(trainFileName, fold);
            String testName = String.format(testFileName, fold);
            Instances trainData = DataSource.read(specificPath + trainName);
            Instances testData = DataSource.read(specificPath + testName);
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);
            System.out.println("Data " + trainName);
            System.out.println("Classifier " + c.getClass().getName());
            c.buildClassifier(trainData);
            if (eval == null)
              eval = new EvaluationPlus(testData);
            else
              eval.setPriors(testData);
            eval.evaluateModel(c, testData);
      } catch (Exception e) {
           System.err.println(e.getMessage());
      }
    }
    System.out.println();
    System.out.println("AUC " + eval.areaUnderROC(positiveLabel));
    System.out.println("GMean " + eval.GMean(positiveLabel));
    System.out.println("Precision " + eval.precision(positiveLabel));
    System.out.println("Recall " + eval.recall(positiveLabel));
    System.out.println("Kappa " + eval.kappa());
    System.out.println("General accuracy " + eval.pctCorrect());
    System.out.println();
  }

  public static void testClassifier_5x5CV() {
    try {
      for (int i = 5; i <= 5; i++) {
        double aveAUC = 0;
        double aveGMean = 0;
        double avePrecision = 0;
        double aveRecall = 0;
        String specificPath = String.format(genericPath, i);
        for (int run = 1; run <= 5; run++) {
          EvaluationPlus eval = null;
          for (int fold = 1; fold <= 5; fold++) {
            String trainName = String.format(trainFileName, i, run, fold);
            String testName = String.format(testFileName, i, run, fold);
            Instances trainData = DataSource.read(specificPath + trainName);
            Instances testData = DataSource.read(specificPath + testName);
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);
            Classifier c = new MIRocchio(positiveLabel, thresholdModel);
            System.out.println("Data " + trainName);
            System.out.println("Classifier " + c.getClass().getName());
            c.buildClassifier(trainData);
            if (eval == null)
              eval = new EvaluationPlus(testData);
            else
              eval.setPriors(testData);
            eval.evaluateModel(c, testData);
          }
          double AUC = eval.areaUnderROC(positiveLabel);
          double GMean = eval.GMean(positiveLabel);
          double Precision = eval.precision(positiveLabel);
          double Recall = eval.recall(positiveLabel);
          aveAUC += AUC;
          aveGMean += GMean;
          avePrecision += Precision;
          aveRecall += Recall;
          System.out.println();
          System.out.println("AUC " + AUC);
          System.out.println("GMean " + GMean);
          System.out.println("Precision " + Precision);
          System.out.println("Recall " + Recall);
          System.out.println();
        }
        aveAUC /= 5;
        aveGMean /= 5;
        avePrecision /= 5;
        aveRecall /= 5;
        System.out.println();
        System.out.println();
        System.out.println("***** Average Performance *****");
        System.out.println();
        System.out.println("Average AUC " + aveAUC);
        System.out.println("Average GMean " + aveGMean);
        System.out.println("Average Precision " + avePrecision);
        System.out.println("Average Recall " + aveRecall);
        System.out.println();
        System.out.println();
      }
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  public static void testClassifier_HoldOut() {
    try {
      for (int i = 1; i <= 3; i++) {
        String trainName = String.format(trainFileName, i, "train");
        String testName = String.format(testFileName, i, "test");
        Instances trainData = DataSource.read(genericPath + trainName);
        Instances testData = DataSource.read(genericPath + testName);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);
        Classifier c = new MIRocchio(positiveLabel, thresholdModel);
        System.out.println("Data " + trainName);
        System.out.println("Classifier " + c.getClass().getName());
        c.buildClassifier(trainData);
        EvaluationPlus eval = new EvaluationPlus(testData);
        eval.evaluateModel(c, testData);
        System.out.println();
        System.out.println("AUC " + eval.areaUnderROC(positiveLabel));
        System.out.println("GMean " + eval.GMean(positiveLabel));
        System.out.println("Precision " + eval.precision(positiveLabel));
        System.out.println("Recall " + eval.recall(positiveLabel));
        System.out.println();
      }
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }
  
  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {
    /**
     * MIRocchio
     */
    Classifier c = new MIRocchio(positiveLabel, thresholdModel);
    /**
     * CitationKNN
     */
    /*
    CitationKNN c = new CitationKNN();
    c.setNumCiters(3);
    c.setNumReferences(3);
    */
    /*
     * MITI
     */
    //MITI c = new MITI();
    /*
     * MILR
     */
    /*
    MILR c = new MILR();
    c.setAlgorithmType(new SelectedTag(MILR.ALGORITHMTYPE_ARITHMETIC, MILR.TAGS_ALGORITHMTYPE));
    */
    /*
     * MISVM
     */
    /*
    MISVM c = new MISVM();
    c.setKernel(new PolyKernel());        // Default: kernel linear
     */
    /**
     * MILES + C4.5
     */
    /*
    FilteredClassifier c = new FilteredClassifier();
    c.setClassifier(new J48());
    c.setFilter(new MILESFilter());
    */

    Launcher.testClassifier_10CV(c);
  }
}
