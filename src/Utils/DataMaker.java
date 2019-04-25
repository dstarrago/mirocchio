/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import java.io.File;
import java.io.FileFilter;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.Instances;
import java.util.Random;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MultiInstanceToPropositional;

/**
 *
 * @author Danel
 */
public class DataMaker {

  private final String path = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/"
          + "_BOOK DATA";
  private final int folds = 5;
  private final int runs = 1;
  private final int seed = 1;

public void makeK_CVPropositional(String fileName) {
  try {
    System.out.println(fileName);
    Instances data = DataSource.read(fileName);
    String output = fileName.substring(fileName.lastIndexOf('\\') + 1, fileName.lastIndexOf('.'));
    File outputDir = new File(path + "/propos_" + output);
    if (!outputDir.exists()) {
      outputDir.mkdir();
    }
    data.setClassIndex(data.numAttributes()-1);
    Random rand = new Random(seed);
    data.randomize(rand);
    data.stratify(folds);
    for (int j = 0; j < folds; j++) {
      Instances train = data.trainCV(folds, j);
      Instances test  = data.testCV(folds, j);
      MultiInstanceToPropositional filter = new MultiInstanceToPropositional();
      filter.setInputFormat(train);
      Instances proposTrain = Filter.useFilter(train, filter);
      DataSink.write(String.format("%s/%s-f%d-train.arff", outputDir.getPath(), output, (j+1)), proposTrain);
      filter.setInputFormat(test);
      Instances proposTest = Filter.useFilter(test, filter);
      DataSink.write(String.format("%s/%s-f%d-test.arff", outputDir.getPath(), output, (j+1)), proposTest);
    }
  } catch (Exception e) {
       System.err.println(e.getMessage());
  }
}

public void makeK_CV(String fileName) {
  try {
    System.out.println(fileName);
    Instances data = DataSource.read(fileName);
    String output = fileName.substring(fileName.lastIndexOf('\\') + 1, fileName.lastIndexOf('.'));
    File outputDir = new File(path + '/' + output);
    if (!outputDir.exists()) {
      outputDir.mkdir();
    }
    data.setClassIndex(data.numAttributes()-1);
    Random rand = new Random(seed);
    data.randomize(rand);
    data.stratify(folds);
    for (int j = 0; j < folds; j++) {
      Instances train = data.trainCV(folds, j);
      Instances test  = data.testCV(folds, j);
      DataSink.write(String.format("%s/%s-f%d-train.arff", outputDir.getPath(), output, (j+1)), train);
      DataSink.write(String.format("%s/%s-f%d-test.arff", outputDir.getPath(), output, (j+1)), test);
    }
  } catch (Exception e) {
       System.err.println(e.getMessage());
  }
}

public void makeTxK_CV(String fileName) {
  try {
    System.out.println(fileName);
    Instances data = DataSource.read(fileName);
    String output = fileName.substring(fileName.lastIndexOf('\\'), fileName.lastIndexOf('.'));
    File outputDir = new File(path + '/' + output);
    if (!outputDir.exists()) {
      outputDir.mkdir();
    }
    data.setClassIndex(data.numAttributes()-1);
    Random rand = new Random(seed);
    for (int i = 0; i < runs; i++) {
      data.randomize(rand);
      data.stratify(folds);
      for (int j = 0; j < folds; j++) {
        Instances train = data.trainCV(folds, j);
        Instances test  = data.testCV(folds, j);
        DataSink.write(String.format("%s/%s-r%d-f%d-train.arff", outputDir.getPath(), output, (i+1), (j+1)), train);
        DataSink.write(String.format("%s/%s-r%d-f%d-test.arff", outputDir.getPath(), output, (i+1), (j+1)), test);
      }
    }
  } catch (Exception e) {
       System.err.println(e.getMessage());
  }
}

private class NotDirectory implements FileFilter {
    public boolean accept(File pathname) {
      return !pathname.isDirectory();
    }
};

public void runMaker(int runs) {
  if (runs < 1) {
    System.out.println("No runs?");
    return;
  }
  // else
  File workingDir = new File(path);
  if (workingDir.exists())
  {
    System.out.println("Path found!");
    NotDirectory notDirectory = new NotDirectory();
    File fileList[] = workingDir.listFiles(notDirectory);
    for (int i = 0; i < fileList.length; i++) {
      if (runs == 1)
          makeK_CV(fileList[i].getPath());
          //makeK_CVPropositional(fileList[i].getPath());
        else
          makeTxK_CV(fileList[i].getPath());
    }
  }
}


/**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
      DataMaker dm = new DataMaker();
      dm.runMaker(dm.runs);
    }

}
