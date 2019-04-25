/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.CostMatrix;

/**
 *
 * @author Danel
 */
public class EvaluationPlus extends Evaluation {

  public EvaluationPlus(Instances data) throws Exception {
    this(data, null);
  }

  public EvaluationPlus(Instances data, CostMatrix costMatrix) throws Exception {
    super(data, null);
  }

  /**
   * Calculate the specificity with respect to a particular class. This
   * is defined as
   * <p/>
   *
   * <pre>
   * correctly classified negatives
   * ------------------------------
   *       total negatives
   * </pre>
   *
   * @param classIndex the index of the class to consider as "positive"
   * @return the specificity
   */
  public double specificity(int classIndex) {
    double TN = 0, total = 0;  // total = TN + FP
    for (int i = 0; i < m_NumClasses; i++) {
      if (i != classIndex) {
        for (int j = 0; j < m_NumClasses; j++) {
          if (j != classIndex) {
            TN += m_ConfusionMatrix[i][j];
          }
          total += m_ConfusionMatrix[i][j];
        }
      }
    }
    if (total == 0) {
      return 0;
    }
    return TN / total;
  }

  /**
   * Calculates the weighted (by class size) specificity.
   *
   * @return the weighted specificity.
   */
  public double weightedSpecificity() {
    double[] classCounts = new double[m_NumClasses];
    double classCountSum = 0;

    for (int i = 0; i < m_NumClasses; i++) {
      for (int j = 0; j < m_NumClasses; j++) {
        classCounts[i] += m_ConfusionMatrix[i][j];
      }
      classCountSum += classCounts[i];
    }

    double specificityTotal = 0;
    for (int i = 0; i < m_NumClasses; i++) {
      double temp = specificity(i);
      specificityTotal += (temp * classCounts[i]);
    }

    return specificityTotal / classCountSum;
  }

  public double GMean(int classIndex) {
    return Math.sqrt(specificity(classIndex) * recall(classIndex));
  }

  /**
   * Calculates the weighted (by class size) GMean.
   *
   * @return the weighted GMean.
   */
  public double weightedGMean() {
    double[] classCounts = new double[m_NumClasses];
    double classCountSum = 0;

    for (int i = 0; i < m_NumClasses; i++) {
      for (int j = 0; j < m_NumClasses; j++) {
        classCounts[i] += m_ConfusionMatrix[i][j];
      }
      classCountSum += classCounts[i];
    }

    double GMeanTotal = 0;
    for (int i = 0; i < m_NumClasses; i++) {
      double temp = GMean(i);
      GMeanTotal += (temp * classCounts[i]);
    }

    return GMeanTotal / classCountSum;
  }

}
