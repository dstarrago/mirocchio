/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package Utils;

import java.util.ArrayList;
import weka.core.Instance;

/**
 * Implements a set of instance in the multi-instances learning framework.
 *
 * @author Dánel S. Tarragó (danels@uclv.edu.cu)
 */
public class InstanceSet {

  /** The items are stored in a vector structure. */
  private ArrayList<Instance> m_instances;
  /** A set of bags is associated to every instance */
  private ArrayList<InstManifestoMI> m_manifestos;
  private boolean useManifestos;
  private int m_posLabel;

  /**
   * Constructs an empty set.
   */
  public InstanceSet() {
    m_instances = new ArrayList<Instance>();
    useManifestos = false;
  }

  /**
   * Constructs an empty set.
   * @param useManif
   */
  public InstanceSet(boolean useManif) {
    m_instances = new ArrayList<Instance>();
    useManifestos = useManif;
    if (useManifestos)
      m_manifestos = new ArrayList<InstManifestoMI>();
  }

  /**
   * Creates a set from the instances of another set.
   *
   * @param s the set containing the instances to be included in the newly created set.
   */
  public InstanceSet(InstanceSet s) {
    this();
    m_instances.addAll(s.m_instances);
    if (s.useManifestos) {
      useManifestos = true;
      m_manifestos = new ArrayList<InstManifestoMI>();
      m_manifestos.addAll(s.m_manifestos);
    }
  }

  public int[] getSortedByPositiveBags() {
    int[] posBagListSizes = new int[m_manifestos.size()];
    for (int i = 0; i < m_manifestos.size(); i++)
      posBagListSizes[i] = getManifesto(i).posBagCount();
    int[] reverse = weka.core.Utils.sort(posBagListSizes);

    int[] index = new int[reverse.length];
    for (int i = 0; i < reverse.length; i++)
      index[i] = reverse[reverse.length -1 - i ];
    return index;
  }

  /**
   * Creates a set from the instances of a bag.
   *
   * @param e the set containing the instances to be included in the newly created set.
   */
  public InstanceSet(Instance e) {
    this();
    for (int i = 0; i < e.relationalValue(1).numInstances(); i++) {
      m_instances.add(e.relationalValue(1).get(i));
    }
  }

  /**
   * Adds an instance to this set.
   *
   * @param i instance to be added.
   */
  public void add(Instance i) {
    m_instances.add(i);
    if (useManifestos) {
      InstManifestoMI M = new InstManifestoMI(m_posLabel);
      m_manifestos.add(M);
    }
  }

  /**
   * Returns <tt>true</tt> if this set contains the specified instance.
   *
   * @param q The instance whose presence in this set is to be tested.
   * @return <tt>true</tt> if this set contains the specified instance.
   */
  public boolean include(Instance q) {
    return indexOf(q) >= 0;
  }

  /**
   * Searches for the first occurence of the given instance.
   *
   * @param q the element to be found
   * @return the index of the first occurrence of the instance in this set;
   * returns -1 if the instance is not found
   */
  public int indexOf(Instance q) {
    for (int i = 0; i < m_instances.size(); i++) {
      Instance p = m_instances.get(i);
      if (Utils.equals(p, q))
        return i;
    }
    return -1;
  }
  
  /**
   * Returns a set containing the intersection between this set and the given set.
   *
   * @param s The set to be intersected.
   * @return a set containing the intersection between this set and the given set.
   */
  public InstanceSet intersec(InstanceSet s) {
    InstanceSet R = new InstanceSet();
    for (int i = 0; i < m_instances.size(); i++) {
      Instance p = m_instances.get(i);
      if (s.include(p))
        R.add(p);
    }
    return R;
  }

  /**
   * Returns a set containing the union between this set and the given set.
   *
   * @param s The set to be united.
   * @return a set containing the union between this set and the given set.
   */
  public InstanceSet union(InstanceSet s) {
    InstanceSet R = new InstanceSet(s);
    for (int i = 0; i < m_instances.size(); i++) {
      Instance p = m_instances.get(i);
      if (!s.include(p))
        R.add(p);
    }
    return R;
  }

  public void setPosLabel(int posLabel) {
    m_posLabel = posLabel;
  }

  public InstManifestoMI getManifesto(int index) {
    return m_manifestos.get(index);
  }

  public InstManifestoMI getLastManifesto() {
    return m_manifestos.get(m_manifestos.size() - 1);     // OJO: ¿es éste el último elemento?
  }

  /**
   * Returns the set current size (i.e. cardinality).
   *
   * @return the set current size.
   */
  public int card() {
    return m_instances.size();
  }

  /**
   * Returns the instance at the given position.
   *
   * @param index the instance's index
   * @return the instance with the given index
   */
  public Instance getInstance(int index) {
    return m_instances.get(index);
  }

}
