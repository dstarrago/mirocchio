/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import java.util.ArrayList;
import weka.core.Instance;

/**
 * Información asociada a cada instancia.
 *
 * @author Dánel
 */
public class InstManifestoMI {
  /** Conjunto de bolsas que contienen a la instancia */
  private ArrayList<Instance> m_bags;
  /** Número de bolsas positivas que contienen a la instancia */
  private int m_posBagCount;
  private int m_posLabel;

  public InstManifestoMI(int thePosLabel) {
    m_bags = new ArrayList<Instance>();
    m_posLabel = thePosLabel;
  }

  public InstManifestoMI(InstManifestoMI p) {
    this(p.m_posLabel);
    m_bags.addAll(p.getBagList());
    m_posBagCount = p.m_posBagCount;
  }

  public int getPosLabel() {
    return m_posLabel;
  }

  public void setPosLabel(int posLabel) {
    m_posLabel = posLabel;
  }

  public void addBag(Instance bag) {
    if (!bagInList(bag)) {
      m_bags.add(bag);
      if (bag.classValue() == m_posLabel)
        m_posBagCount++;
    }
  }

  private boolean bagInList(Instance bag) {
    return bagIndex(bag) >= 0;
  }

  private int bagIndex(Instance bag) {
    for (int i = 0; i < m_bags.size(); i++) {
      Instance p = m_bags.get(i);
      if (p == bag)
        return i;
    }
    return -1;
  }

  public ArrayList<Instance> getBagList() {
    return m_bags;
  }

  public int posBagCount() {
    return m_posBagCount;
  }

  public int negBagCount() {
    return m_bags.size() - m_posBagCount;
  }

  public int bagCount() {
    return m_bags.size();
  }

  public void appendPosBags(InstManifestoMI M) {
    ArrayList<Instance> bagList = M.getBagList();
    for (int i = 0; i < bagList.size(); i++) {
      Instance bag = bagList.get(i);
      if (bag.classValue() == m_posLabel)
        addBag(bag);
    }
  }

  public void appendBags(InstManifestoMI M) {
    ArrayList<Instance> bagList = M.getBagList();
    for (int i = 0; i < bagList.size(); i++) {
      addBag(bagList.get(i));
    }
  }
}
