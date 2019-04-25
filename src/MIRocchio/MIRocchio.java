/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package MIRocchio;

import Utils.InstManifestoMI;
import Utils.InstanceSet;
import Utils.EvaluationPlus;
import Utils.Similarity;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;
import java.util.ArrayList;
//import weka.core.SparseInstance;

/**
 * Classificador multinstancia Rocchio.
 * 
 * Este es el mismo MIRocchio5 con formato relacional en lugar de usar la clase milk.Exemplar
 *
 * La imputación de las instancias es la siguiente: si la instancia aparece en
 * alguna bolsa negativa entonces se asigna a la clase negativa, en caso contrario
 * se asigna a la clase positiva. Las instancias repetidas se toman una sola vez.
 * Los parámetros ro+ y ro- son determinados automáticamente a partir de los
 * datos de entrenamiento.
 * 
 * @author Danel
 */
public class MIRocchio extends AbstractClassifier {

  /**
   * Índice de la etiqueta de clase positiva
   */
  private int m_posLabel;
  /**
   * Conjunto de ejemplos de entrenamiento
   */
  protected Instances m_data;
  /**
   * Conjunto de instancias extraidas de las bolsas de entrenamiento
   * InstanceSet tiene información de las bolsas que contienen a cada instancia.
   */
  protected InstanceSet m_instances;
  /**
   * Listado de instancias positivas
   */
  private ArrayList<Instance> posInst;
  /**
   * Listado de instancias negativas
   */
  private ArrayList<Instance> negInst;
  /**
   * Perfil positivo
   */
  private double posWeight[];
  /**
   * Perfil negativo
   */
  private double negWeight[];
  /**
   * Umbral que determina cuando la bolsa es positiva
   */
  private double bagTh;
  /**
   * Lista de posibles valores del parámetro ro+ para buscar el óptimo
   */
  private double[] posRoVals = new double[] {15, 20, 25, 30, 40};
  /**
   * Lista de posibles valores del parámetro ro- para buscar el óptimo
   */
  private double[] negRoVals = new double[] {2, 3, 4, 8, 15, 20};
  /**
   * Valor óptimo del parámetro ro+
   */
  private double posRo;
  /**
   * Valor óptimo del parámetro ro-
   */
  private double negRo;
  /**
   * Lista de movimientos que iterativamente se aplicará en la búsqueda del
   * valor óptimo de ro+
   */
  private int[] movePos = new int[] { -1, 0, 1, 1, 1, 0, -1, -1};
  /**
   * Lista de movimientos que iterativamente se aplicará en la búsqueda del
   * valor óptimo de ro-
   */
  private int[] moveNeg = new int[] { -1, -1, -1, 0, 1, 1, 1, 0};
  /**
   * Bandera para activar la traza, con fines documentales
   */
  private boolean trace;
  /**
   * Centroide positivo, con fines documentales
   */
  private double posCenter[];
  /**
   * Centroide negativo, con fines documentales
   */
  private double negCenter[];
  /**
   * Centro absoluto, con fines documentales
   */
  //private double center[];

  private Stats testPosStats;
  private Stats testNegStats;

  /**
   * Threshold models
   */
  public static final int tm_Gaussian = 1;
  public static final int tm_Linear = 2;
  /**
   * Modelo seleccionado para el cálculo del umbral
   */
  private int thresholdModel;
  
  public MIRocchio(int poslabel, int theTresholdModel) {
    m_posLabel = poslabel;
    thresholdModel = theTresholdModel;
  }

  @Override
  public void buildClassifier(Instances exs) throws Exception {
    m_data = exs;
    testPosStats = new Stats();
    testNegStats = new Stats();
    trace = false;
    findInstances();
    imputeInstances();
    //findParameters();
    //trace = true;
    posRo = 1;
    negRo = 1;
    findWeights(posRo, negRo);      // el mejor valor para Musk 1 es 30, 2
    findThreshold();
    //System.out.println("bagTh = " + bagThreshold());
  }

  private void findParameters() throws Exception {
    // Índice del primer mejor valor de Ro+ representado como double
    double iniPos = 0;
    // Índice del último mejor valor de Ro+ representado como double
    double endPos = 0;
    // Índice del primer mejor valor de Ro- representado como double
    double iniNeg = 0;
    // Índice del último mejor valor de Ro- representado como double
    double endNeg = 0;
    // Mejor AUC al evaluar el desempeño del algoritmo
    double betterPerf = 0;
    // Matriz con los desempeños de la clasificación en entrenamiento con
    // cada combinación de los parámetros ro+ y ro-
    double[][] perfMatrix = new double[negRoVals.length][posRoVals.length];
    if (trace) {
      System.out.println();
      System.out.print(String.format("%6.0f  ", 0.0));
      for (int j = 0; j < posRoVals.length; j++) {
        System.out.print(String.format("%6.0f  ", posRoVals[j]));
      }
      System.out.println();
    }
    for (int i = 0; i < negRoVals.length; i++) {
      if (trace)
        System.out.print(String.format("%6.0f  ", negRoVals[i]));
      for (int j = 0; j < posRoVals.length; j++) {
        findWeights(posRoVals[j], negRoVals[i]);
        findThreshold();
        EvaluationPlus e = new EvaluationPlus(m_data);
        e.evaluateModel(this, m_data);
        double performance = e.pctCorrect();
        perfMatrix[i][j] = performance;
        if (trace)
          System.out.print(String.format("%6.4f  ", performance));
        if (performance > betterPerf) {
          iniPos = endPos = j;
          iniNeg = endNeg = i;
          betterPerf = performance;
        } else
          if (performance == betterPerf) {
            endPos = j;
            endNeg = i;
            if (j < iniPos)
              iniPos = j;
          }
      }
      if (trace)
        System.out.println();
    }
    if (trace)
      System.out.println();
    //System.out.println("iniPos " + iniPos);
    //System.out.println("endPos " + endPos);
    //System.out.println("iniNeg " + iniNeg);
    //System.out.println("endNeg " + endNeg);
    int centralPosIndex = (int)Math.ceil(iniPos + (endPos - iniPos) / 2.0);
    int centralNegIndex = (int)Math.ceil(iniNeg + (endNeg - iniNeg) / 2.0);
    int pIndex = centralPosIndex;
    int nIndex = centralNegIndex;
    for (int attempt = 0; perfMatrix[nIndex][pIndex] < betterPerf && attempt < movePos.length; attempt++) {
      pIndex = centralPosIndex + movePos[attempt];
      if (pIndex < 0 || pIndex > posRoVals.length) {
        pIndex = centralPosIndex;
        continue;
      }
      nIndex = centralNegIndex + moveNeg[attempt];
      if (nIndex < 0 || nIndex > negRoVals.length) {
        nIndex = centralNegIndex;
        continue;
      }
    }
    if (perfMatrix[nIndex][pIndex] < betterPerf) {
      pIndex = (int) iniPos;
      nIndex = (int) iniNeg;
    }
    //System.out.println("posIndex " + pIndex);
    //System.out.println("negIndex " + nIndex);
    posRo = posRoVals[pIndex];
    negRo = negRoVals[nIndex];
    if (trace) {
      System.out.println("Ro+ óptimo " + posRo);
      System.out.println("Ro- óptimo " + negRo);
    }
  }

  /**
   * Extrae las instancias de las bolsas y las coloca en el campo m_instances
   */
  private void findInstances() {
    m_instances = new InstanceSet(true);
    m_instances.setPosLabel(m_posLabel);
    for (int i = 0; i < m_data.numInstances(); i++) {
      Instance B = m_data.get(i);
      Instances I = B.relationalValue(1);
      for (int j = 0; j < I.numInstances(); j++) {
        Instance p = I.instance(j);
        int index = m_instances.indexOf(p);
        if (index >= 0) {
          /// La instancia ya se había encontrado anteriormente y está en la posición "index"
          InstManifestoMI M = m_instances.getManifesto(index);
          M.addBag(B);
        } else {
          // La instancia aparece por primera vez
          m_instances.add(p);
          InstManifestoMI M = m_instances.getLastManifesto();
          M.addBag(B);
        }
      }
    }
  }

  /**
   * Realiza el proceso de imputación de etiquetas de clase a las instancias.
   * La imputación de las instancias es la siguiente: si la instancia aparece en
   * alguna bolsa negativa entonces se asigna a la clase negativa, en caso contrario
   * se asigna a la clase positiva.
   */
  private void imputeInstances() {
    posInst = new ArrayList<Instance>();
    negInst = new ArrayList<Instance>();
    for (int i = 0; i < m_instances.card(); i++) {
      InstManifestoMI m = m_instances.getManifesto(i);
      if (m.negBagCount() > 0) {
        negInst.add(m_instances.getInstance(i));
      } else {
        posInst.add(m_instances.getInstance(i));
      }
    }
  }

  /**
   * Calcula el peso de cada atributo en los perfiles de clase.
   *
   * @param posRo parámetro que controla la cantidad de selección de atributo a
   * realizar en el perfil positivo. Un número mayor implica mayor selección de
   * atributo.
   *
   * @param negRo parámetro que controla la cantidad de selección de atributo a
   * realizar en el perfil negativo. Un número mayor implica mayor selección de
   * atributo.
   *
   * @throws java.lang.Exception
   */
  private void findWeights(double posRo, double negRo)  throws Exception {
    final int bagNumAttrs = m_data.attribute(1).relation().numAttributes();
    //boolean sparse = m_data.get(0).relationalValue(1).instance(0) instanceof SparseInstance;
    posWeight = new double[bagNumAttrs];
    negWeight = new double[bagNumAttrs];
    if (trace) {
      posCenter = new double[bagNumAttrs];
      negCenter = new double[bagNumAttrs];
      //center = new double[bagNumAttrs];
    }
    int zeroPos = 0;
    int zeroNeg = 0;
    int nonZeroPos = 0;
    int nonZeroNeg = 0;
    for (int i = 0; i < bagNumAttrs; i++) {
      double posAggregate = 0, negAggregate = 0;
      for (int j = 0; j < posInst.size(); j++) {
        Instance d = posInst.get(j);
        posAggregate += d.value(i);
      }
      double posAverage = posAggregate / posInst.size();
      for (int j = 0; j < negInst.size(); j++) {
        Instance d = negInst.get(j);
        negAggregate += d.value(i);
      }
      double negAverage = negAggregate / negInst.size();
      double posDifference = posAverage - posRo * negAverage;
      double negDifference = negAverage - negRo * posAverage;
      /*
      if (trace) {
        if (posDifference > 0) {
          //System.out.println(String.format("Wi+ = %4.4f - %2.1f * %4.4f = %4.4f", posAverage, posRo, negAverage, posDifference));
          System.out.println("Wi+ " + negAverage);
          nonZeroPos++;
        } else {
          zeroPos++;
        }
        if (negDifference > 0) {
          //System.out.println(String.format("Wi- = %4.4f - %2.1f * %4.4f = %4.4f", negAverage, negRo, posAverage, negDifference));
          System.out.println("Wi- " + posAverage);
          nonZeroNeg++;
        } else {
          zeroNeg++;
        }
      }
       */
      if (trace) {
        posCenter[i] = posAverage;
        negCenter[i] = negAverage;
        //center[i] = (posAggregate + negAggregate) / (posInst.size() + negInst.size());
      }
      if (posDifference < 0)
        posWeight[i] = 0;
      else
        posWeight[i] = posDifference;
      if (negDifference < 0)
        negWeight[i] = 0;
      else
        negWeight[i] = negDifference;
    }
    /*
    if (trace) {
      System.out.println();
      System.out.println("Wi+ es cero " + zeroPos + " veces y distinto de cero " + nonZeroPos + " veces");
      System.out.println("Wi- es cero " + zeroNeg + " veces y distinto de cero " + nonZeroNeg + " veces");
      System.out.println();
    }
     *
     */
  }

  /**
   * Calcula la proporción de instancias positivas dentro de una bolsa.
   *
   * @param exemplar bolsa a la que se desea calcular la proporción.
   * @return número entre 0 y 1 que indica la proporción.
   * @throws java.lang.Exception
   */
  private double bagEstimation(Instance instance) throws Exception {
    int posCount = 0;
    for (int i = 0; i < instance.relationalValue(1).numInstances(); i++) {
      Instance x = instance.relationalValue(1).instance(i);
      if (Similarity.cos(x, posWeight) > Similarity.cos(x, negWeight))
        posCount++;
    }
    return (double)posCount / instance.relationalValue(1).numInstances();
  }

  /**
   * Calcula el umbral de proporción a partir del cual una bolsa se considera positiva.
   * @return valor del umbral de proporción.
   * @throws java.lang.Exception
   */
  private double findThreshold() throws Exception {
    Stats posStats = new Stats();
    Stats negStats = new Stats();
    for (int i = 0; i < m_data.numInstances(); i++) {
      double gb = bagEstimation(m_data.get(i));
      if (m_data.get(i).classValue() == m_posLabel) {
        posStats.add(gb);
      } else {
        negStats.add(gb);
      }
    }
    posStats.calculateDerived();
    negStats.calculateDerived();
    double Mp = posStats.mean;
    double Mn = negStats.mean;
    double Dp = posStats.stdDev;
    double Dn = negStats.stdDev;
    if (trace) {
      System.out.println("Estadisticas en entrenamiento ");
      System.out.println("Media pos: " + Mp);
      System.out.println("StdDv pos: " + Dp);
      System.out.println("Media neg: " + Mn);
      System.out.println("StdDv neg: " + Dn);
    }
    if (thresholdModel == MIRocchio.tm_Linear) {
      bagTh = (Mp * Dn + Mn * Dp) / (Dp + Dn);                                        // ponderada lineal
      if (trace) {
        System.out.println();
        System.out.println("Umbral lineal en entrenamiento " + bagTh);
        System.out.println();
      }
    } else {
      // thresholdModel == MIRocchio5.tm_Gaussian
      double eta = 10;
      double dp2 = Dp * Dp;
      double dn2 = Dn * Dn;
      double gp = Math.exp(-dp2 * eta);
      double gn = Math.exp(-dn2 * eta);
      bagTh = (Mp * gp + Mn * gn) / (gp + gn);                                        // Gaussian
      if (trace) {
        System.out.println();
        System.out.println("Umbral exponencial en entrenamiento " + bagTh);
        System.out.println();
      }
    }
    return bagTh;
  }

  public void resetStats() {
    testPosStats = new Stats();
    testNegStats = new Stats();
  }

  public void printStats() {
    testPosStats.calculateDerived();
    testNegStats.calculateDerived();
    double Mp = testPosStats.mean;
    double Mn = testNegStats.mean;
    double Dp = testPosStats.stdDev;
    double Dn = testNegStats.stdDev;
    System.out.println("Test Media pos: " + Mp);
    System.out.println("Test StdDv pos: " + Dp);
    System.out.println("Test Media neg: " + Mn);
    System.out.println("Test StdDv neg: " + Dn);
    System.out.println("Test positivos: " + testPosStats.count);
    System.out.println("Test negativos: " + testNegStats.count);

    //double eta = 10.0 * pos / neg;
    double eta = 10;
    double Vp = Dp * Dp;
    double Vn = Dn * Dn;
    double gp = Math.exp(-Vp * eta);
    double gn = Math.exp(-Vn * eta);
    double num = (Mp * Dn + Mn * Dp);
    double den = (Dp + Dn);
    double linealTh = num / den;
    //double linealTh = (Mp * Dn + Mn * Dp) / (Dp + Dn);                                        // ponderada lineal
    double expTh = (Mp * gp + Mn * gn) / (gp + gn);                                        // Gaussian
    System.out.println();
    System.out.println("Umbral lineal en test " + linealTh);
    System.out.println("Umbral exponencial en test " + expTh);
    System.out.println();
  }

  /**
   * Clasifica un ejemplo (bolsa) según la siguiente regla: si la proporción de
   * instancias positivas de la bolsa excede un cierto umbral (calculado en el entrenamiento)
   * la intancia es positiva, en otro caso es negativa.
   *
   * @param exemplar bolsa a clasificar.
   * @return Clase de la bolsa en formato double.
   * @throws java.lang.Exception
   */
  private double thresholdClassification(Instance instance)  throws Exception {
    double m = bagEstimation(instance);
    if (trace) {
      if (instance.classValue() == m_posLabel) {
        testPosStats.add(m);
      } else {
        testNegStats.add(m);
        //System.out.println(gb);
      }
    }
    if (m >= bagTh) {
      /*
      if (exemplar.classValue() == 0) {
        System.out.println("FP " + m);
      }*/
      return 1;
    }
    else {
      /*
      if (exemplar.classValue() == 1) {
        System.out.println("FN " + m);
      }*/
      return 0;
    }
  }

  /**
   * Clasifica un ejemplo (bolsa).
   *
   * @param instance bolsa a clasificar.
   * @return Clase de la bolsa en formato double.
   * @throws java.lang.Exception
   */
  @Override
  public double classifyInstance(Instance instance) throws Exception {
    return thresholdClassification(instance);
  }

}
