/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import weka.core.FastVector;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.SparseInstance;

/**
 * Class implementing some simple utility methods.
 *
 * @author Dánel
 */
public class Utils {

  /**
   * Check whether two instances have equal attibutes values.
   *
   * The method assumes that both instance have the
   * same attributes, they don't have to belong to the same dataset.
   *
   * @param a the first instance to compare
   * @param b the second instance to compare
   * @return true if instance a equal to instance b; false otherwise.
   */
  
  static public boolean equals(Instance a, Instance b) {
    if (a instanceof SparseInstance)        // Asumo que b tambien es SparseInstance
      return SparseEquals((SparseInstance)a, (SparseInstance)b);

    boolean equal = true;
    for (int i = 0; i < a.numAttributes() && equal; i++) {
      // 1. special handling if missing value (NaN) is involved:
      if (a.isMissing(i) || b.isMissing(i)) {
        if (a.isMissing(i) && b.isMissing(i))
          continue;
      } else equal = false;
      // 2. regular values:
      equal = weka.core.Utils.eq(a.value(i), b.value(i));
    }
    return equal;
  }

  static public boolean SparseEquals(SparseInstance a, SparseInstance b) {
    int aLength = a.numValues();
    int bLength = b.numValues();
    if (aLength != bLength) return false;
    boolean equal = true;

    for (int i = 0; i < aLength && equal; i++) {
      if (a.index(i) != b.index(i)) return false;
      // 1. special handling if missing value (NaN) is involved:
      if (((Instance)a).isMissing(i) || ((Instance)b).isMissing(i)) {
        if (((Instance)a).isMissing(i) && ((Instance)b).isMissing(i))
          continue;
        else
          return false;
      } 
      // 2. regular values:
      equal = weka.core.Utils.eq(a.valueSparse(i), b.valueSparse(i));
    }
    return equal;
  }

  static public double LukasiewiczImp(double a, double b) {
    double c = 1 - a + b;
    return (c < 1)? c: 1;
  }

  static public double Kleene_DienesImp(double a, double b) {
    double c = 1 - a;
    return (c > b)? c : b;
  }

  static public double LukasiewiczTNorm(double a, double b) {
    double c = a + b - 1;
    return (c > 0)? c : 0;
  }

  static public double minimumTNorm(double a, double b) {
    return (a < b)? a: b;
  }
  
  static public double quartil(int c, double[] x) {
    if (x.length == 0)
      return 0;
    double d = c * x.length / 4.0;
    int i = (int)d;
    if (d == (double)i)
      return (x[i] + x[i-1])/2;
    else
      return x[i];
  }

  /**
   * Calcula el percentil p-ésimo de un arreglo de números.
   * @param p percentil deseado
   * @param x arreglo de números. 
   * @return
   */
  static public double percentil(double p, double[] x) {
    if (x.length == 0)
      return 0;
    double d = p * x.length;
    int i = (int)d;
    if (d == (double)i)
      return (x[i] + x[i-1])/2;
    else
      return x[i];
  }

  static public double separabilityQ(FastVector posVal, FastVector negVal) {
    double c = 0;
    for (int i = 0; i < posVal.size(); i++)  {
      double pv = (Double)posVal.elementAt(i);
      for (int j = 0; j < negVal.size(); j++)  {
        double nv = (Double)negVal.elementAt(j);
        if (pv > nv)
          c++;
        else
          c--;
      }
    }
    return c / (posVal.size() * negVal.size());
  }

  static public double separabilityR(FastVector posVal, FastVector negVal) {
    double s = 0;
    double z = 0;
    for (int i = 0; i < posVal.size(); i++)  {
      double pv = (Double)posVal.elementAt(i);
      for (int j = 0; j < negVal.size(); j++)  {
        double nv = (Double)negVal.elementAt(j);
        double d = (pv - nv);
        s += d;
        z += d * d;
      }
    }
    return s / Math.sqrt(z);
  }

  static public double separabilityS(FastVector upper, FastVector lower) {
    double s = 0;
    double z = 0;
    for (int i = 0; i < upper.size(); i++)  {
      double pv = (Double)upper.elementAt(i);
      for (int j = 0; j < lower.size(); j++)  {
        double nv = (Double)lower.elementAt(j);
        double d = (pv - nv);
        s += d;
        if (d >= 0)
          z += d;
        else
          z += -d;
      }
    }
    return s / z;
  }

}
