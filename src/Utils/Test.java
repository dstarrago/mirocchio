/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import java.util.Random;

/**
 *
 * @author Danel
 */
public class Test {

  private static final double gamma = 40;
  private static final double dev = 5;
  private static final int N = 10;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
      System.out.println("Gamma = " + gamma);
      Random R = new Random();
      double x;
      double Gave = 0;
      for (int i = 0; i < N; i++) {
        x = R.nextGaussian();
        double g = gamma + x * dev;
        System.out.println("G = " + g);
        Gave += g;
      }
      Gave /= N;
      System.out.println("G promedio = " + Gave);
    }

}
