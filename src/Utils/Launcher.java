/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.Instances;
import weka.filters.Filter;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;

/**
 *
 * @author Danel
 */
public class Launcher {

  final static private String problemName = "WIR9";
  final static private String workName = "tfc-tfc-V9";

  final static private String genericPath = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/34 " + problemName + "/";
  final static private String fileName = workName + ".arff";
  final static private String outputName = problemName + "Sel.arff";

  public static void selectAttr() {
    try {
      Instances dataset = DataSource.read(genericPath + fileName);
      dataset.setClassIndex(dataset.numAttributes() - 1);
      System.out.println("Procesando " + problemName);
      RemoveRare removeRare = new RemoveRare();
      removeRare.setInputFormat(dataset);
      Instances outputData = Filter.useFilter(dataset, removeRare);
      DataSink.write(genericPath + outputName, outputData);
      System.out.println("¡Completo! " + problemName);
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
      selectAttr();
    }

}
