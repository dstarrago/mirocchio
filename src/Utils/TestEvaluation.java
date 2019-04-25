/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.classifiers.Evaluation;
import weka.classifiers.mi.CitationKNN;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Danel
 */
public class TestEvaluation {

  final static private String fileName = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/_Colección MIL #1/"
          + "01 Musk1/Musk1.arff";
  
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
      try {
        /**
         * Cargar los datos de entrenamiento y prueba (en este ejemplo se toman los mismos datos)
         */
        Instances trainData = DataSource.read(fileName);
        Instances testData = DataSource.read(fileName);

        /**
         * Asignar la clase a ambos conjuntos de datos
         */
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);
        /**
         * Asignar peso 1 a los ejemplos (bolsas) de entrenamiento
         */
        for (int i = 0; i < testData.numInstances(); i++) 
          testData.instance(i).setWeight(1);
        /**
         * Construir y configurar el clasificador
         */
        CitationKNN c = new CitationKNN();
        c.setNumCiters(3);
        c.setNumReferences(3);
        /**
         * Reportar configuración del experimento
         */
        System.out.println("Data " + trainData.relationName());
        System.out.println("Classifier " + c.getClass().getName());
        /**
         * Entrenar
         */
        c.buildClassifier(trainData);
        /**
         * Evaluar el clasificador entrenado en los datos de prueba
         */
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(c, testData);
        /**
         * Presentar los resultados de la evaluación
         */
        System.out.println(eval.toSummaryString());
      } catch (Exception e) {
           System.err.println(e.getMessage());
      }
    }

}
