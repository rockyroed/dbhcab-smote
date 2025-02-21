/*  Downloaded from http://grepcode.com/file/repo1.maven.org/maven2/nz.ac.waikato.cms.weka/CABSMOTEsII/1.0.1/weka/filters/supervised/instance/CABSMOTEsII.java 
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * CABSMOTEsII.java
 * 
 * Copyright (C) 2008 Ryan Lichtenwalter 
 * Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 */

package weka.filters.supervised.instance;

import weka.experiment.Stats;
import weka.core.AttributeStats;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.core.Capabilities.Capability;

import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.WeightedInstancesHandler;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import java.util.Arrays;
import java.util.IntSummaryStatistics;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import weka.clusterers.SimpleKMeans;

import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;
import javax.swing.*;

/**
 <!-- globalinfo-start -->
 * 01 Resamples a dataset by applying the Synthetic Minority Oversampling TEchnique (CABSMOTEsII). The original dataset must fit entirely in memory. The amount of CABSMOTEsII and number of nearest neighbors may be specified. For more information, see <br/>
 * <br/>
 * Nitesh V. Chawla et. al. (2002). Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research. 16:321-357.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{al.2002,
 *    author = {Nitesh V. Chawla et. al.},
 *    journal = {Journal of Artificial Intelligence Research},
 *    pages = {321-357},
 *    title = {Synthetic Minority Over-sampling Technique},
 *    volume = {16},
 *    year = {2002}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -S &lt;num&gt;
 *  Specifies the random number seed
 *  (default 1)</pre>
 * 
 * <pre> -P &lt;percentage&gt;
 *  Specifies percentage of CABSMOTEsII instances to create.
 *  (default 50.0)
 * </pre>
 * 
 * <pre> -K &lt;nearest-neighbors&gt;
 *  Specifies the number of nearest neighbors to use.
 *  (default 5)
 * </pre>
 * 
 * <pre> -C &lt;value-index&gt;
 *  Specifies the index of the nominal class value to CABSMOTEsII
 *  (default 0: auto-detect non-empty minority class))
 * </pre>
 * 
 <!-- options-end -->
 *  
 * @author Ryan Lichtenwalter (rlichtenwalter@gmail.com)
 * @version $Revision: 8108 $
 */
public class CABSMOTEsII extends Filter  implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {

  /** for serialization. */
  static final long serialVersionUID = -1653880819059250364L;

//  public int DangerToClass= 0;

  public String DTC= "sample";
   
  /** the number that determine the gap range 0 tends to minority, 1 tends to majority*/ 
  public double GapRange=1; 
  
  /** the number that determine the Kmeans used for clustering, defualt= 2 */ 
  public int KmeansValue= 2;  
  
  /** the number of M neighbors to use. */
  protected int m_NearestNeighbors = 5;
  
  /** the number of M neighbors to use. */
  protected int k_NearestNeighbors = 5;
  
  /** the random seed to use. */
  protected int m_RandomSeed = 1;
  
  /** the percentage of CABSMOTEsII instances to create. */
  protected double m_Percentage = 2.0;
  
  /** the index of the class value. */
  protected String m_ClassValueIndex = "0";
  
  /** whether to detect the minority class automatically. */
  protected boolean m_DetectMinorityClass = true;

  /**
   * Returns a string describing this classifier.
   * 
   * @return 		a description of the classifier suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Resamples a dataset by applying the Synthetic Minority Oversampling TEchnique (CABSMOTEsII)." +
    " The original dataset must fit entirely in memory." +
    " The amount of CABSMOTEsII and number of nearest neighbors may be specified." +
    " For more information, see \n\n" 
    + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return 		the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);

    result.setValue(Field.AUTHOR, "Nitesh V. Chawla et. al.");
    result.setValue(Field.TITLE, "Synthetic Minority Over-sampling Technique");
    result.setValue(Field.JOURNAL, "Journal of Artificial Intelligence Research");
    result.setValue(Field.YEAR, "2002");
    result.setValue(Field.VOLUME, "16");
    result.setValue(Field.PAGES, "321-357");

    return result;
  }

  /**
   * Returns the revision string.
   * 
   * @return 		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8108 $");
  }

  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector newVector = new Vector();
    
    newVector.addElement(new Option(
	"\tSpecifies the random number seed\n"
	+ "\t(default 1)",
	"S", 1, "-S <num>"));
    
    newVector.addElement(new Option(
	"\tSpecifies percentage of CABSMOTEsII instances to create.\n"
	+ "\t(default 100.0)\n",
	"P", 1, "-P <percentage>"));
	
	newVector.addElement(new Option(
	"\tSpecifies Kmeans value used in clustering.\n"
	+ "\t(default 2)\n",
	"E", 1, "-E <KmeansValue>"));
    
    newVector.addElement(new Option(
	"\tSpecifies the number M of nearest neighbors to use between minority and total instances to create the danger subset.\n"
	+ "\t(default 5)\n",
	"M", 1, "-M <MNearest-Neighbors>"));
	
	newVector.addElement(new Option(
	"\tSpecifies the number of nearest neighbors to use between danger instance and other instances.\n"
	+ "\t(default 5)\n",
	"K", 1, "-K <KNearest-Neighbors>"));
    
    newVector.addElement(new Option(
	"\tSpecifies the index of the nominal class value to CABSMOTEsII\n"
	+"\t(default 0: auto-detect non-empty minority class))\n",
	"C", 1, "-C <value-index>"));

    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -S &lt;num&gt;
   *  Specifies the random number seed
   *  (default 1)</pre>
   * 
   * <pre> -P &lt;percentage&gt;
   *  Specifies percentage of CABSMOTEsII instances to create.
   *  (default 100.0)
   * </pre>
   * 
   * <pre> -K &lt;nearest-neighbors&gt;
   *  Specifies the number of nearest neighbors to use.
   *  (default 5)
   * </pre>
   * 
   * <pre> -C &lt;value-index&gt;
   *  Specifies the index of the nominal class value to CABSMOTEsII
   *  (default 0: auto-detect non-empty minority class))
   * </pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String seedStr = Utils.getOption('S', options);
    if (seedStr.length() != 0) {
      setRandomSeed(Integer.parseInt(seedStr));
    } else {
      setRandomSeed(1);
    }

    String percentageStr = Utils.getOption('P', options);
    if (percentageStr.length() != 0) {
      setPercentage(new Double(percentageStr).doubleValue());
    } else {
      setPercentage(100.0);
    }

    String KmeansStr = Utils.getOption('E', options);
	if (KmeansStr.length() != 0) {
      setKmeansValue(Integer.parseInt(KmeansStr));
    } else {
      setKmeansValue(2);
    }
	
	String nnStr1 = Utils.getOption('G', options);
	if (nnStr1.length() != 0) {
	      setGapRange(Integer.parseInt(nnStr1));
	    } else {
	      setGapRange(1);
	    }
		
/*	 String DTCStr = Utils.getOption('D', options);
	 if (DTCStr.length() != 0) {
	      setDangerToClass(Integer.parseInt(DTCStr));
	    } else {
	      setDangerToClass(0);
	    } */

    String nnStr = Utils.getOption('M', options);
    if (nnStr.length() != 0) {
      setNearestNeighbors(Integer.parseInt(nnStr));
    } else {
      setNearestNeighbors(5);
    }
	
	String knnStr = Utils.getOption('K', options);
    if (knnStr.length() != 0) {
      setKNearestNeighbors(Integer.parseInt(knnStr));
    } else {
      setKNearestNeighbors(5);
    }

    String classValueIndexStr = Utils.getOption( 'C', options);
    if (classValueIndexStr.length() != 0) {
      setClassValue(classValueIndexStr);
    } else {
      m_DetectMinorityClass = true;
    }
  }

  /**
   * Gets the current settings of the filter.
   *
   * @return an array 	of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector<String>	result;
    
    result = new Vector<String>();
    
    result.add("-C");
    result.add(getClassValue());
	
	result.add("-GR");
	result.add("" + getGapRange());
	
/*  result.add("-D2C");
	result.add("" + getDangerToClass());*/
	
    result.add("-M");
    result.add("" + getNearestNeighbors());
	
    result.add("-E");
    result.add("" + getKmeansValue());	
	
	result.add("-K");
    result.add("" + getKNearestNeighbors());
    
    result.add("-P");
    result.add("" + getPercentage());
    
    result.add("-S");
    result.add("" + getRandomSeed());
    
    return result.toArray(new String[result.size()]);
  }

  /**
   * Returns the tip text for this property.
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String randomSeedTipText() {
    return "The seed used for random sampling.";
  }

  /**
   * Gets the random number seed.
   *
   * @return 		the random number seed.
   */
  public int getRandomSeed() {
    return m_RandomSeed;
  }

  /**
   * Sets the random number seed.
   *
   * @param value 	the new random number seed.
   */
  public void setRandomSeed(int value) {
    m_RandomSeed = value;
  }

  /**
   * Returns the tip text for this property.
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
   
  public String KmeansValueTipText() {
	    return "the number of Kmeans clusters.";
	  }
	  
  public void setKmeansValue(int value){
	if ( value >0)
		KmeansValue = value;
	else
		System.err.println("value should be larger than 0");

	}
	
	public int getKmeansValue()
	{
		return KmeansValue;
	}

   
  public String percentageTipText() {
    return "The percentage of CABSMOTEsII instances to create. 1.5 adds 50%, 2 adds 100%, 3 adds 200%, 4 adds 300%,5 adds 400%";
  }

  /**
   * Sets the percentage of CABSMOTEsII instances to create.
   * 
   * @param value	the percentage to use
   */
  public void setPercentage(double value) {
    if (value >= 0)
      m_Percentage = value;
    else
      System.err.println("Percentage must be >= 0!");
  }

  /**
   * Gets the percentage of CABSMOTEsII instances to create.
   * 
   * @return 		the percentage of CABSMOTEsII instances to create
   */
  public double getPercentage() {
    return m_Percentage;
  }

  /**
   * Returns the tip text for this property.
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String nearestNeighborsTipText() {
    return "Number of M nearest neighbors to use between minority and total instances to create the danger subset.";
  }
  
    public String knearestNeighborsTipText() {
    return "Number of nearest neighbors to use between danger instance and other instances.";
  }

  /**
   * Sets the number of nearest neighbors to use.
   * 
   * @param value	the number of nearest neighbors to use
   */
  public void setNearestNeighbors(int value) {
    if (value >= 1)
      m_NearestNeighbors = value;
    else
      System.err.println("At least 1 neighbor necessary!");
  }

  /**
   * Gets the number of nearest neighbors to use.
   * 
   * @return 		the number of nearest neighbors to use
   */
  public int getNearestNeighbors() {
    return m_NearestNeighbors;
  }
  
    
  public void setKNearestNeighbors(int value) {
    if (value >= 1)
      k_NearestNeighbors = value;
    else
      System.err.println("At least 1 neighbor necessary!");
  }

  /**
   * Gets the number of nearest neighbors to use.
   * 
   * @return 		the number of nearest neighbors to use
   */
  public int getKNearestNeighbors() {
    return k_NearestNeighbors;
  }

  /**
   * Returns the tip text for this property.
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String classValueTipText() {
    return "The index of the class value to which CABSMOTEsII should be applied. " +
    "Use a value of 0 to auto-detect the non-empty minority class.";
  }

  /**
   * Sets the index of the class value to which CABSMOTEsII should be applied.
   * 
   * @param value	the class value index
   */
  public void setClassValue(String value) {
    m_ClassValueIndex = value;
    if (m_ClassValueIndex.equals("0")) {
      m_DetectMinorityClass = true;
    } else {
      m_DetectMinorityClass = false;
    }
  }

  /**
   * Gets the index of the class value to which CABSMOTEsII should be applied.
   * 
   * @return 		the index of the clas value to which CABSMOTEsII should be applied
   */
  public String getClassValue() {
    return m_ClassValueIndex;
  }

  /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo 	an Instances object containing the input 
   * 				instance structure (any instances contained in 
   * 				the object are ignored - only the structure is required).
   * @return 			true if the outputFormat may be collected immediately
   * @throws Exception 		if the input format can't be set successfully
   */
  public boolean setInputFormat(Instances instanceInfo) throws Exception {
    super.setInputFormat(instanceInfo);
    super.setOutputFormat(instanceInfo);
    return true;
  }

  /**
   * Input an instance for filtering. Filter requires all
   * training instances be read before producing output.
   *
   * @param instance 		the input instance
   * @return 			true if the filtered instance may now be
   * 				collected with output().
   * @throws IllegalStateException if no input structure has been defined
   */
  public boolean input(Instance instance) {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }
    if (m_FirstBatchDone) {
      push(instance);
      return true;
    } else {
      bufferInput(instance);
      return false;
    }
  }

    public String GapRangeTipText() {
	    return "The number  determine the gap range 0 tends to minority, 1 tends to other class.";
	  }
  	public void setGapRange(double value){
		if (value >= 0 && value <= 1)
			GapRange = value;
		else
			System.err.println("value should be between 0 and 1!");
	}
	
		public double getGapRange()
	{
		return GapRange;
	}
    
/*      public String DangerToClassTipText() {
	    return "Check NN between danger and: 0 minority, 1 majority, 2 Total ,3 Danger.";
	  }
  	public void setDangerToClass(int value1){
		
		if (value1 >= 0 && value1 <= 3)
		{	DangerToClass = value1;
			if ( DangerToClass ==0)
				DTC="sample";
			if ( DangerToClass ==1)
				DTC="Majsample";
			if ( DangerToClass ==2)
				DTC="total";
			if ( DangerToClass ==3)
				DTC="danger";             
		}
		else
			System.err.println("value should be between 0 and 3!");	

	}
	
			public int getDangerToClass()
	{
		return DangerToClass;
	}  */
  /**
   * Signify that this batch of input to the filter is finished. 
   * If the filter requires all instances prior to filtering,
   * output() may now be called to retrieve the filtered instances.
   *
   * @return 		true if there are instances pending output
   * @throws IllegalStateException if no input structure has been defined
   * @throws Exception 	if provided options cannot be executed 
   * 			on input instances
   */
  public boolean batchFinished() throws Exception {
    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    if (!m_FirstBatchDone) {
      // Do CABSMOTEsII, and clear the input instances.
      doCABSMOTEsII();
	  	 // JOptionPane.showMessageDialog(null,"CABSMOTEsII 19-09-2018 ");
		 
    }
    flushInput();

    m_NewBatch = true;
    m_FirstBatchDone = true;
    return (numPendingOutput() != 0);
  }

  /**
   * The procedure implementing the CABSMOTEsII algorithm. The output
   * instances are pushed onto the output queue for collection.
   * 
   * @throws Exception 	if provided options cannot be executed 
   * 			on input instances
   */
  protected void doCABSMOTEsII() throws Exception 
  {
    int minIndex = 0;
    int min = Integer.MAX_VALUE;
    if (m_DetectMinorityClass) 
	{
	  // find minority class
      int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
      for (int i = 0; i < classCounts.length; i++) 
	  {
		if (classCounts[i] != 0 && classCounts[i] < min) 
		{
			min = classCounts[i];
			minIndex = i;
		}
      }
    } else 
	{
     String classVal = getClassValue();
     if (classVal.equalsIgnoreCase("first")) 
	  {
		minIndex = 1;
      } else if (classVal.equalsIgnoreCase("last")) 
			{
			minIndex = getInputFormat().numClasses();
			} else 
				{
				minIndex = Integer.parseInt(classVal);
				}
     if (minIndex > getInputFormat().numClasses()) 
		{
		throw new Exception("value index must be <= the number of classes");
		}
      minIndex--; // make it an index
    }

    int nearestNeighbors;
    if (min <= getNearestNeighbors()) {
      nearestNeighbors = min - 1;
    } else {
      nearestNeighbors = getNearestNeighbors();
    }
    if (nearestNeighbors < 1)
      throw new Exception("Cannot use 0 neighbors!");
  
       int knearestNeighbors;
    if (min <= getKNearestNeighbors()) {
      knearestNeighbors = min - 1;
    } else {
      knearestNeighbors = getKNearestNeighbors();
    }
    if (knearestNeighbors < 1)
      throw new Exception("Cannot use 0 neighbors!");
 
    // compose minority class dataset
    // also push all dataset instances
    Instances sample = getInputFormat().stringFreeStructure();
	Instances danger = getInputFormat().stringFreeStructure();
	Instances total  = getInputFormat().stringFreeStructure();
	Instances Majsample = getInputFormat().stringFreeStructure();
    Enumeration instanceEnum = getInputFormat().enumerateInstances();
    while(instanceEnum.hasMoreElements()) {
      Instance instance = (Instance) instanceEnum.nextElement();
      push((Instance) instance.copy());
	  total.add(instance);	  
      if ((int) instance.classValue() == minIndex) {
	sample.add(instance);
      } else
	  { Majsample.add(instance);
	  }
	  
    }

    // compute Value Distance Metric matrices for nominal features
    Map vdmMap = new HashMap();
    Enumeration attrEnum = getInputFormat().enumerateAttributes();
    while(attrEnum.hasMoreElements()) {
      Attribute attr = (Attribute) attrEnum.nextElement();
      if (!attr.equals(getInputFormat().classAttribute())) {
	if (attr.isNominal() || attr.isString()) {
	  double[][] vdm = new double[attr.numValues()][attr.numValues()];
	  vdmMap.put(attr, vdm);
	  int[] featureValueCounts = new int[attr.numValues()];
	  int[][] featureValueCountsByClass = new int[getInputFormat().classAttribute().numValues()][attr.numValues()];
	  instanceEnum = getInputFormat().enumerateInstances();
	  while(instanceEnum.hasMoreElements()) {
	    Instance instance = (Instance) instanceEnum.nextElement();
	    int value = (int) instance.value(attr);
	    int classValue = (int) instance.classValue();
	    featureValueCounts[value]++;
	    featureValueCountsByClass[classValue][value]++;
	  }
	  for (int valueIndex1 = 0; valueIndex1 < attr.numValues(); valueIndex1++) {
	    for (int valueIndex2 = 0; valueIndex2 < attr.numValues(); valueIndex2++) {
	      double sum = 0;
	      for (int classValueIndex = 0; classValueIndex < getInputFormat().numClasses(); classValueIndex++) {
		double c1i = (double) featureValueCountsByClass[classValueIndex][valueIndex1];
		double c2i = (double) featureValueCountsByClass[classValueIndex][valueIndex2];
		double c1 = (double) featureValueCounts[valueIndex1];
		double c2 = (double) featureValueCounts[valueIndex2];
		double term1 = c1i / c1;
		double term2 = c2i / c2;
		sum += Math.abs(term1 - term2);
	      }
	      vdm[valueIndex1][valueIndex2] = sum;
	    }
	  }
	}
      }
    }

    // use this random source for all required randomness
    Random rand = new Random(getRandomSeed());

    // find the set of extra indices to use if the percentage is not evenly divisible by 100
    List extraIndices = new LinkedList();
    double percentageRemainder = (100 / 100) - Math.floor(100.0 / 100.0);
    int extraIndicesCount = (int) (percentageRemainder * sample.numInstances());
    if (extraIndicesCount >= 1) {
      for (int i = 0; i < sample.numInstances(); i++) {
	extraIndices.add(i);
      }
    }
    Collections.shuffle(extraIndices, rand);
    extraIndices = extraIndices.subList(0, extraIndicesCount);
    Set extraIndexSet = new HashSet(extraIndices);

    // the main loop to handle computing nearest neighbors and generating CABSMOTEsII
    // examples from each instance in the original minority class data
	// Instance[] nnArray = new Instance[nearestNeighbors];
	// Instance[] mmcounter = new Instance[nearestNeighbors]; // mm stands for number of instance in majority class
  for (int i = 0; i < sample.numInstances(); i++)
	{
		Instance instanceI = sample.instance(i);
		// find k nearest neighbors for each instance
		List distanceToInstance = new LinkedList();
		for (int j = 0; j < total.numInstances(); j++) 
		{
			Instance instanceJ = total.instance(j);
			if (i != j) 
			{
				double distance = 0;
				attrEnum = getInputFormat().enumerateAttributes();
				while(attrEnum.hasMoreElements()) 
				{
					Attribute attr = (Attribute) attrEnum.nextElement();
					if (!attr.equals(getInputFormat().classAttribute())) 
					{
						double iVal = instanceI.value(attr);
						double jVal = instanceJ.value(attr);
						if (attr.isNumeric()) 
						{
						distance += Math.pow(iVal - jVal, 2);
						} else 
							{
							distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
							}
					}
				}
	  distance = Math.pow(distance, .5);
	  distanceToInstance.add(new Object[] {distance, instanceJ});
			}
		}
      // sort the neighbors according to distance
      Collections.sort(distanceToInstance, new Comparator() 
	  {
		public int compare(Object o1, Object o2) 
		{
			double distance1 = (Double) ((Object[]) o1)[0];
			double distance2 = (Double) ((Object[]) o2)[0];
			return Double.compare(distance1, distance2);
		  //return (int) Math.ceil(distance1 - distance2);
		}
	   });
	    
      // populate the actual nearest neighbor instance array
      Iterator entryIterator = distanceToInstance.iterator();
      int j=0;
	  int mm=0; // m' ( number of majority samples)
  

      while(entryIterator.hasNext() && j < nearestNeighbors) 
	  {

		Instance inst = (Instance) ((Object[])entryIterator.next())[1];
		int clsI = (int) inst.classValue();
		if (clsI != minIndex)
			mm++;
		j++;
      }
	// Creation of danger set
	  if ( mm >= (nearestNeighbors/2) && mm < nearestNeighbors)
			{
				danger.add(instanceI);
			}
    }
	
    
	int tempclassIndex = danger.classIndex();
//	JOptionPane.showMessageDialog(null," First Danger class Index = " + tempclassIndex);
	danger.setClassIndex(-1) ;  //  unset the class before you pass the data to kmeans.
//	JOptionPane.showMessageDialog(null,"After SetClass to -1 = " + danger.classIndex());

	SimpleKMeans kmeans = new SimpleKMeans();
	kmeans.setNumClusters(KmeansValue);
	kmeans.setPreserveInstancesOrder(true);
	kmeans.buildClusterer(danger);
	weka.core.Instances[] dataset = new weka.core.Instances[kmeans.getNumClusters()]; 
		for (int i = 0; i < dataset.length; i++) 
			{
				dataset[i] = new Instances(danger, 0);
			}

		for (Instance inst : danger)
			{
				dataset[(int)kmeans.clusterInstance(inst)].add(inst);
			}
		
	danger.setClassIndex(tempclassIndex) ;
//	JOptionPane.showMessageDialog(null," After giving it the old value ="+ danger.classIndex());
	
		for (int i=0; i<kmeans.getNumClusters();i++)
		{
//			JOptionPane.showMessageDialog(null," ClassIndex before ="+ dataset[i].classIndex());
			dataset[i].setClassIndex(tempclassIndex);
//			JOptionPane.showMessageDialog(null," ClassIndex after ="+ dataset[i].classIndex());
		}
		
		Double per[] = new Double[kmeans.getNumClusters()];
	
		for (int i=0; i<kmeans.getNumClusters();i++)
		{
		per[i]=(double) dataset[i].numInstances() / (double)danger.numInstances(); 
    //  JOptionPane.showMessageDialog(null," Percentage of dataset ["+ i +"]="+ (per[i])*100 );	
//	JOptionPane.showMessageDialog(null," Number of instances of dataset ["+ i +"]="+ (dataset[i]).numInstances() );
		}
//  http://zparacha.com/minimum-maximum-array-value-java		
	int permaxValueindex = 0;	
	double permaxValue = per[0];
		  for(int i=0;i < per.length;i++)
		  {
			if(per[i] > permaxValue)
			{
				permaxValue = per[i];
				permaxValueindex=i;
			}
		  }
//	JOptionPane.showMessageDialog(null," Maximum Percentage index ["+ permaxValueindex +"]="+ (permaxValue));	  
	
	int perminValueindex = 0;	
	double perminValue = per[0];
		  for(int i=0;i < per.length;i++)
		  {
			if(per[i] < perminValue)
			{
				perminValue = per[i];
				perminValueindex=i;
			}
		  }	
		 Double perdatatomax[] = new Double[kmeans.getNumClusters()]; 
		 for (int i=0; i<kmeans.getNumClusters();i++)
		{
		perdatatomax[i]=(double) dataset[i].numInstances() / (double)dataset[permaxValueindex].numInstances(); 
 //     JOptionPane.showMessageDialog(null," Percentage of dataset ["+ i +"]="+ (perdatatomax[i])*100 );	
//	 JOptionPane.showMessageDialog(null," Number of instances of dataset ["+ i +"]="+ (dataset[i]).numInstances() );
		}
//	JOptionPane.showMessageDialog(null," Minimum Percentage index ["+ perminValueindex +"]="+ (perminValue));
//	JOptionPane.showMessageDialog(null," Minimum Percentage number of instance ="+ dataset[perminValueindex].numInstances());
//	JOptionPane.showMessageDialog(null," Maximum Percentage number of instance ="+ dataset[permaxValueindex].numInstances());	
//	JOptionPane.showMessageDialog(null," Percentage of dataset ["+ 0 +"]="+ (per[0])*100 );
	
Instance[] nnArray = new Instance[knearestNeighbors];	
	
	for ( int a=0;a<per.length;a++)
	{
		int n= dataset[permaxValueindex].numInstances();
		int l=dataset[a].numInstances();
    //	if (a != permaxValueindex )	
		if (perdatatomax[a] >= 0.50 )
		{
		//	for (int b= dataset[a].numInstances();b< dataset[permaxValueindex].numInstances();b++)
		//	{
	        //JOptionPane.showMessageDialog(null," instances number ="+ b);
			//	JOptionPane.showMessageDialog(null," If statement is true"+perdatatomax[a]);
			
			for (int i = 0; i < dataset[a].numInstances(); i++)
				{
					Instance instanceI = dataset[a].instance(i);
					// find k nearest neighbors for each instance
					List distanceToInstance = new LinkedList();
					for (int j = 0; j < dataset[a].numInstances(); j++)
					{
					Instance instanceJ = dataset[a].instance(j);
					if (i != j) 
						{
						double distance = 0;
						attrEnum = getInputFormat().enumerateAttributes();
						while(attrEnum.hasMoreElements()) 
						{
							Attribute attr = (Attribute) attrEnum.nextElement();
							if (!attr.equals(getInputFormat().classAttribute())) 
								{
								double iVal = instanceI.value(attr);
								double jVal = instanceJ.value(attr);
								if (attr.isNumeric()) 
									{
									distance += Math.pow(iVal - jVal, 2);
									} else 
									{
									distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
									}
								}
						}
						distance = Math.pow(distance, .5);
						distanceToInstance.add(new Object[] {distance, instanceJ});
						}
						
					}
					
						Collections.sort(distanceToInstance, new Comparator() 
						{
						public int compare(Object o1, Object o2) 
						{
							double distance1 = (Double) ((Object[]) o1)[0];
							double distance2 = (Double) ((Object[]) o2)[0];
							return Double.compare(distance1, distance2);
							//return (int) Math.ceil(distance1 - distance2);
						}});
						
						// populate the actual nearest neighbor instance array  nnArray between danger (I) and Majoritysample 
						Iterator entryIterator = distanceToInstance.iterator();
						int z=0;
						while(entryIterator.hasNext() && z < knearestNeighbors) 
						{
						nnArray[z] = (Instance) ((Object[])entryIterator.next())[1];
						z++;
						}
						// create synthetic examples
			//	for (int b= dataset[a].numInstances();b< dataset[permaxValueindex].numInstances();b++)
						
						
						while(l < (dataset[a].numInstances()*  m_Percentage) || extraIndexSet.remove(i)) 
						{
							
							double[] values = new double[sample.numAttributes()];   
							int nn = (rand.nextInt(knearestNeighbors));
							attrEnum = getInputFormat().enumerateAttributes();
							while(attrEnum.hasMoreElements()) 
							{
								Attribute attr = (Attribute) attrEnum.nextElement();
								if (!attr.equals(getInputFormat().classAttribute())) 
								{
								if (attr.isNumeric()) {
								double dif = nnArray[nn].value(attr) - instanceI.value(attr);
								double gap = rand.nextDouble()%GapRange;
								values[attr.index()] = (double) (instanceI.value(attr) + gap * dif);
								} else if (attr.isDate()) {
								double dif = nnArray[nn].value(attr) - instanceI.value(attr);
								double gap = rand.nextDouble()%GapRange;
								values[attr.index()] = (long) (instanceI.value(attr) + gap * dif);
								} else 
								{
								int[] valueCounts = new int[attr.numValues()];
								int iVal = (int) instanceI.value(attr);
								valueCounts[iVal]++;
								for (int nnEx = 0; nnEx < knearestNeighbors; nnEx++) 
								{
									int val = (int) nnArray[nnEx].value(attr);
									valueCounts[val]++;
								}
								int maxIndex = 0;
								int max = Integer.MIN_VALUE;
								for (int index = 0; index < attr.numValues(); index++) 
								{
									if (valueCounts[index] > max) 
									{
									max = valueCounts[index];
									maxIndex = index;
									}
								}
									values[attr.index()] = maxIndex;
								}
								}
							}	
	  
						values[sample.classIndex()] = minIndex;
						Instance synthetic = new DenseInstance(1.0, values);
						push(synthetic);
						
						
						
					//	JOptionPane.showMessageDialog(null," a number ="+ a +" n number ="+ n);
						l++;

						}		
				
				}
			
		//	}
				
		}
		
	}

	
  }
	

  /**
   * Main method for running this filter.
   *
   * @param args 	should contain arguments to the filter: 
   * 			use -h for help
   */
  public static void main(String[] args) {
    runFilter(new CABSMOTEsII(), args);
  }
}
