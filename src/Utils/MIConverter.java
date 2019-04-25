/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.Instances;
//import weka.filters.unsupervised.attribute.PropositionalToMultiInstance;
import weka.filters.Filter;
import weka.core.SparseInstance;

/**
 *
 * @author Danel
 */
public class MIConverter {
  
  static private final String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/TREC9 (OHSUMED)/1/AttrsFixed/";
  static private final String nameMask = "TREC9-1-10-%d%s.arff";    // %d = 1..10, %s = tst, tra
  static private final String name = "TREC9-1fixed.arff";
  static private final int folds = 10;

  static private void convertFile() {
    try {
      Instances data = DataSource.read(path + name);
      data.setClassIndex(data.numAttributes() - 1);
      System.out.println("Converting " + name + " to relational format");
      if (data.instance(0) instanceof SparseInstance)
        System.out.println("It brings sparse instances!");
      PropToMultiInst propToMIL = new PropToMultiInst();
      propToMIL.setInputFormat(data);
      System.out.println("Input format done!");
      Instances MIData = Filter.useFilter(data, propToMIL);
      String outputFile = String.format("%s/MI_%s", path, name);
      DataSink.write(outputFile, MIData);
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  static private void convert(int fold, String role) {
    try {
      String trainName = String.format(nameMask, fold, role);
      Instances data = DataSource.read(path + trainName);
      data.setClassIndex(data.numAttributes() - 1);
      System.out.println("Converting " + trainName + " to relational format");
      if (data.instance(0) instanceof SparseInstance)
        System.out.println("It brings sparse instances!");
      PropToMultiInst propToMIL = new PropToMultiInst();
      propToMIL.setInputFormat(data);
      System.out.println("Input format done!");
      Instances MIData = Filter.useFilter(data, propToMIL);
      String outputFile = String.format("%s/XMI_%s", path, trainName);
      DataSink.write(outputFile, MIData);
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  static private void convertPar() {
    for (int fold = 1; fold <= folds; fold++) {
      convert(fold, "tra");
      break;
      //convert(fold, "tst");
    }
  }

  /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
      convertFile();
    }

}
