/*  Downloaded from http://grepcode.com/file/repo1.maven.org/maven2/nz.ac.waikato.cms.weka/HCABSMOTE/1.0.1/weka/filters/supervised/instance/HCABSMOTE.java
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
 * DBHCABSMOTE.java
 *
 * Copyright (C) 2008 Ryan Lichtenwalter
 * Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 */

package weka.filters.supervised.instance;

import weka.clusterers.OPTICS;
import weka.clusterers.forOPTICSAndDBScan.DataObjects.DataObject;
import weka.clusterers.forOPTICSAndDBScan.Databases.Database;
import weka.core.Capabilities.Capability;

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
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import weka.filters.unsupervised.instance.RemoveRange;

import java.util.*;
import javax.swing.*;
import javax.xml.crypto.Data;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.clusterers.DBSCAN;
import weka.filters.SimpleBatchFilter;

/**
 * <!-- globalinfo-start -->
 * 01 Resamples a dataset by applying the Synthetic Minority Oversampling TEchnique (DBHCABSMOTE). The original dataset must fit entirely in memory. The amount of DBHCABSMOTE and number of nearest neighbors may be specified. For more information, see <br/>
 * <br/>
 * Nitesh V. Chawla et. al. (2002). Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research. 16:321-357.
 * <p/>
 * <!-- globalinfo-end -->
 * <p>
 * <!-- technical-bibtex-start -->
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
 * <!-- technical-bibtex-end -->
 * <p>
 * <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -S &lt;num&gt;
 *  Specifies the random number seed
 *  (default 1)</pre>
 *
 * <pre> -P &lt;percentage&gt;
 *  Specifies percentage of DBHCABSMOTE instances to create.
 *  (default 100.0)
 * </pre>
 *
 * <pre> -K &lt;nearest-neighbors&gt;
 *  Specifies the number of nearest neighbors to use.
 *  (default 5)
 * </pre>
 *
 * <pre> -C &lt;value-index&gt;
 *  Specifies the index of the nominal class value to DBHCABSMOTE
 *  (default 0: auto-detect non-empty Minority class))
 * </pre>
 * <p>
 * <!-- options-end -->
 *
 * @author Ryan Lichtenwalter (rlichtenwalter@gmail.com)
 * @version $Revision: 8108 $
 */
public class DBHCABSMOTE extends Filter implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {
    /**
     * for serialization.
     */
    static final long serialVersionUID = -1653880819059250364L;

    /**
     * the number that determine the gap range 0 tends to Minority, 1 tends to majority
     */
    public double GapRange = 1;

    /**
     * the number of M neighbors to use.
     */
    protected int m_NearestNeighbors = 5;

    /**
     * the number of K neighbors to use.
     */
    protected int k_NearestNeighbors = 5;

    /**
     * the random seed to use.
     */
    protected int m_RandomSeed = 1;

    /**
     * the percentage of DBHCABSMOTE instances to create.
     */
    protected double m_Percentage = 2.0;

    /**
     * the index of the class value.
     */
    protected String m_ClassValueIndex = "0";

    /**
     * whether to detect the Minority class automatically.
     */
    protected boolean m_DetectMinorityClass = true;

    /**
     * the value of epsilon (radius of the circular range)
     */
    public double epsilon = 1.0;

    /**
     * the value of minPts (minimum number of points to consider a cluster)
     */
    public int minPts = 6;


    /**
     * the mapping of the instances that will be used for clustering
     */
    public Database database;
    private final HashMap<Instance, DataObject> instanceDataObjectMapping = new HashMap<Instance, DataObject>();
    private final HashMap<Instance, Instance> instanceMapping = new HashMap<Instance, Instance>();

    /**
     * Returns a string describing this classifier.
     *
     * @return a description of the classifier suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Resamples a dataset by applying the Density-based Hybrid Clustered Affinitive Borderline Synthetic Minority Oversampling Technique (DBHCABSMOTE)." +
                "Uses the original architecture of HCABSMOTE, but changed the clustering technique from Kmeans clustering to Density-based Clustering of Applications with Noise (DBSCAN)" +
                " The original dataset must fit entirely in memory." +
                " The amount of DBHCABSMOTE and number of nearest neighbors may be specified." +
                " For more information, see \n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
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
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8108 $");
    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
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
     * Parses a given list of options.
     * <p>
     * <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -S &lt;num&gt;
     *  Specifies the random number seed
     *  (default 1)</pre>
     *
     * <pre> -P &lt;percentage&gt;
     *  Specifies percentage of DBHCABSMOTE instances to create.
     *  (default 100.0)
     * </pre>
     *
     * <pre> -K &lt;nearest-neighbors&gt;
     *  Specifies the number of nearest neighbors to use.
     *  (default 5)
     * </pre>
     *
     * <pre> -C &lt;value-index&gt;
     *  Specifies the index of the nominal class value to DBHCABSMOTE
     *  (default 0: auto-detect non-empty Minority class))
     * </pre>
     * <p>
     */

//    @Override
//    protected Instances determineOutputFormat(Instances inputFormat) {
//        return inputFormat;
//    }
//
//    @Override
//    protected Instances process(Instances instances) {
//        return instances;
//    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {
        Vector<Option> options = new Vector<>();

        options.addElement(new Option(
                "\tSpecifies the number M of nearest neighbors to use between Minority and total instances to create the mindanger subset.\n"
                        + "\t(default " + m_NearestNeighbors + ")\n",
                "MN", 1, "-MN <mNearest-Neighbors>"));

        options.addElement(new Option(
                "\tThe number that determines the gap range. 0 tends to Minority, 1 tends to majority.\n"
                        + "\t(default " + GapRange + ")\n",
                "G", 1, "-G <gap-range>"));

        options.addElement(new Option(
                "\tEpsilon parameter for DBSCAN.\n"
                        + "\t(default " + epsilon + ")\n",
                "E", 1, "-E <epsilon>"));

        options.addElement(new Option(
                "\tMinPoints parameter for DBSCAN.\n"
                        + "\t(default " + minPts + ")\n",
                "M", 1, "-M <minPoints>"));

        options.addElement(new Option(
                "\tSpecifies the number of nearest neighbors to use between mindanger instance and other instances.\n"
                        + "\t(default " + k_NearestNeighbors + ")\n",
                "K", 1, "-K <KNearest-Neighbors>"));

        options.addElement(new Option(
                "\tSpecifies the index of the nominal class value to HCABSMOTE\n"
                        + "\t(default 0: auto-detect non-empty Minority class))\n",
                "C", 1, "-C <value-index>"));

        options.addElement(new Option(
                "\tPercentage of SMOTE instances to create.\n"
                        + "\t(default  " + m_Percentage + ")\n",
                "P", 1, "-P <percentage>"));

        options.addElement(new Option(
                "\tRandom seed for SMOTE.\n"
                        + "\t(default  " + m_RandomSeed + ")\n",
                "S", 1, "-S <seed>"));

        return options.elements();
    }

    /**
     * Parses a given list of options.
     * <p>
     * <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -S &lt;num&gt;
     *  Specifies the random number seed
     *  (default 1)</pre>
     *
     * <pre> -P &lt;percentage&gt;
     *  Specifies percentage of DBHCABSMOTE instances to create.
     *  (default 100.0)
     * </pre>
     *
     * <pre> -K &lt;nearest-neighbors&gt;
     *  Specifies the number of nearest neighbors to use.
     *  (default 5)
     * </pre>
     *
     * <pre> -C &lt;value-index&gt;
     *  Specifies the index of the nominal class value to DBHCABSMOTE
     *  (default 0: auto-detect non-empty Minority class))
     * </pre>
     * <p>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        String mNearestStr = Utils.getOption("MN", options);
        if (!mNearestStr.isEmpty()) {
            setNearestNeighbors(Integer.parseInt(mNearestStr));
        } else {
            setNearestNeighbors(m_NearestNeighbors);
        }

        String gapRangeStr = Utils.getOption('G', options);
        if (!gapRangeStr.isEmpty()) {
            setGapRange(Double.parseDouble(gapRangeStr));
        } else {
            setGapRange(GapRange);
        }

        String epsilonStr = Utils.getOption("E", options);
        if (!epsilonStr.isEmpty()) {
            setEpsilon(Double.parseDouble(epsilonStr));
        } else {
            setEpsilon(epsilon);
        }

        String minPtsStr = Utils.getOption("M", options);
        if (!minPtsStr.isEmpty()) {
            setMinPts(Integer.parseInt(minPtsStr));
        } else {
            setMinPts(minPts);
        }

        String kNearestStr = Utils.getOption("K", options);
        if (!kNearestStr.isEmpty()) {
            setKNearestNeighbors(Integer.parseInt(kNearestStr));
        } else {
            setKNearestNeighbors(k_NearestNeighbors);
        }

        String classValueIndexStr = Utils.getOption('C', options);
        if (!classValueIndexStr.isEmpty()) {
            setClassValue(classValueIndexStr);
        } else {
            m_DetectMinorityClass = true;
        }

        String percentageStr = Utils.getOption('P', options);
        if (!percentageStr.isEmpty()) {
            setPercentage(Double.parseDouble(percentageStr));
        } else {
            setPercentage(m_Percentage);
        }

        String seedStr = Utils.getOption('S', options);
        if (!seedStr.isEmpty()) {
            setRandomSeed(Integer.parseInt(seedStr));
        } else {
            setRandomSeed(m_RandomSeed);
        }
    }

    /**
     * Gets the current settings of the filter.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        Vector<String> options = new Vector<>();

        options.add("-MN");
        options.add("" + m_NearestNeighbors);

        options.add("-G");
        options.add("" + GapRange);

        options.add("-E");
        options.add("" + epsilon);

        options.add("-M");
        options.add("" + minPts);

        options.add("-K");
        options.add("" + k_NearestNeighbors);

        options.add("-C");
        options.add(m_ClassValueIndex);

        options.add("-P");
        options.add("" + m_Percentage);

        options.add("-S");
        options.add("" + m_RandomSeed);

        return options.toArray(new String[0]);
    }

    /**
     * Gets the number of nearest neighbors to use.
     *
     * @return the number of nearest neighbors to use
     */
    public int getNearestNeighbors() {
        return m_NearestNeighbors;
    }

    /**
     * Sets the number of nearest neighbors to use.
     *
     * @param value the number of nearest neighbors to use
     */
    public void setNearestNeighbors(int value) {
        if (value >= 1)
            m_NearestNeighbors = value;
        else
            System.err.println("At least 1 M nearest neighbor necessary.");
    }

    /**
     * Gets the gap range.
     *
     * @return the gap range value
     */
    public double getGapRange() {
        return GapRange;
    }

    /**
     * Sets the gap range
     *
     * @param value the gap range value
     */
    public void setGapRange(double value) {
        if (value >= 0 && value <= 1)
            GapRange = value;
        else
            System.err.println("Gap range should be between 0 and 1.");
    }

    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Sets the epsilon for DBSCAN
     *
     * @param value the epsilon value for DBSCAN
     */
    public void setEpsilon(double value) {
        if (value > 0)
            epsilon = value;
        else
            System.err.println("Epsilon should be greater than 0.");
    }

    public int getMinPts() {
        return minPts;
    }

    /**
     * Sets the minimum points for clustering in DBSCAN
     *
     * @param value the minimum points value for DBSCAN
     */
    public void setMinPts(int value) {
        if (value >= 1)
            minPts = value;
        else
            System.err.println("Minimum points should be greater than or equal to 1.");
    }

    /**
     * Gets the number of nearest neighbors to use.
     *
     * @return the number of nearest neighbors to use
     */
    public int getKNearestNeighbors() {
        return k_NearestNeighbors;
    }


    /**
     * Sets the number of K nearest neighbors to use.
     *
     * @param value the number of K nearest neighbors to use
     */
    public void setKNearestNeighbors(int value) {
        if (value >= 1)
            k_NearestNeighbors = value;
        else
            System.err.println("At least 1 K nearest neighbor necessary.");
    }

    /**
     * Gets the index of the class value to which DBHCABSMOTE should be applied.
     *
     * @return the index of the clas value to which DBHCABSMOTE should be applied
     */
    public String getClassValue() {
        return m_ClassValueIndex;
    }

    /**
     * Sets the index of the class value to which DBHCABSMOTE should be applied.
     *
     * @param value the class value index
     */
    public void setClassValue(String value) {
        m_ClassValueIndex = value;
        m_DetectMinorityClass = m_ClassValueIndex.equals("0");
    }

    /**
     * Gets the random number seed.
     *
     * @return the random number seed.
     */
    public int getRandomSeed() {
        return m_RandomSeed;
    }

    /**
     * Sets the random number seed.
     *
     * @param value the new random number seed.
     */
    public void setRandomSeed(int value) {
        m_RandomSeed = value;
    }

    /**
     * Gets the percentage of DBHCABSMOTE instances to create.
     *
     * @return the percentage of DBHCABSMOTE instances to create
     */
    public double getPercentage() {
        return m_Percentage;
    }

    /**
     * Sets the percentage of DBHCABSMOTE instances to create.
     *
     * @param value the percentage to use
     */
    public void setPercentage(double value) {
        if (value > 0)
            m_Percentage = value;
        else
            System.err.println("Percentage must be greater than 0.");
    }

    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input
     *                     instance structure (any instances contained in
     *                     the object are ignored - only the structure is required).
     * @return true if the outputFormat may be collected immediately
     * @throws Exception if the input format can't be set successfully
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
     * @param instance the input instance
     * @return true if the filtered instance may now be
     * collected with output().
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

    /**
     * Signify that this batch of input to the filter is finished.
     * If the filter requires all instances prior to filtering,
     * output() may now be called to retrieve the filtered instances.
     *
     * @return true if there are instances pending output
     * @throws IllegalStateException if no input structure has been defined
     * @throws Exception             if provided options cannot be executed
     *                               on input instances
     */
    public boolean batchFinished() throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (!m_FirstBatchDone) {
            // Do DBHCABSMOTE, and clear the input instances.
            doDBHCABSMOTE();

        }
        flushInput();

        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }

    // Get the cluster label of each instance
    public int clusterInstance(Instance instance) throws Exception {
        Instance filteredInstance = instanceMapping.get(instance);
        DataObject dataObject = (DataObject) instanceDataObjectMapping.get(filteredInstance);
        if (dataObject.getClusterLabel() == DataObject.NOISE)
            return -2;
        else {
            return dataObject.getClusterLabel();
        }
    }

    /**
     * The procedure implementing the DBHCABSMOTE algorithm. The output
     * instances are pushed onto the output queue for collection.
     *
     * @throws Exception if provided options cannot be executed
     *                   on input instances
     */
    protected void doDBHCABSMOTE() throws Exception {
        /**
         * Prints the parameter input for checking
         */
        System.out.println("\n\nM Nearest Neighbors: " + m_NearestNeighbors);
        System.out.println("Gap Range: " + GapRange);
        System.out.println("Epsilon: " + epsilon);
        System.out.println("Minimum Points: " + minPts);
        System.out.println("K Nearest Neighbors: " + k_NearestNeighbors);
        System.out.println("Class Value Index: " + m_ClassValueIndex);
        System.out.println("Percentage: " + m_Percentage);
        System.out.println("Random Seed: " + m_RandomSeed);

        int minIndex = 0;
        int min = Integer.MAX_VALUE;
        if (m_DetectMinorityClass) {
            // find Minority class
            int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
            for (int i = 0; i < classCounts.length; i++) {
                if (classCounts[i] != 0 && classCounts[i] < min) {
                    min = classCounts[i];
                    minIndex = i;
                }
            }
        } else {
            String classVal = getClassValue();
            if (classVal.equalsIgnoreCase("first")) {
                minIndex = 1;
            } else if (classVal.equalsIgnoreCase("last")) {
                minIndex = getInputFormat().numClasses();
            } else {
                minIndex = Integer.parseInt(classVal);
            }
            if (minIndex > getInputFormat().numClasses()) {
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

        // compose Minority class dataset
        // also push all dataset instances
        Instances Minority = getInputFormat().stringFreeStructure();
        Instances mindanger = getInputFormat().stringFreeStructure();
        Instances total = getInputFormat().stringFreeStructure();
        Instances Majority = getInputFormat().stringFreeStructure();
        Instances Newtotal = getInputFormat().stringFreeStructure();
        Instances MajNonDanger = getInputFormat().stringFreeStructure(); // majority without mindanger
        Instances Generated = getInputFormat().stringFreeStructure(); // new genereated instances
        Instances MinNonDanger = getInputFormat().stringFreeStructure(); // Minority without mindanger
        Enumeration instanceEnum = getInputFormat().enumerateInstances();
        while (instanceEnum.hasMoreElements()) {
            Instance instance = (Instance) instanceEnum.nextElement();
            push((Instance) instance.copy());
            total.add(instance);
            if ((int) instance.classValue() == minIndex) {
                Minority.add(instance);
            } else {
                Majority.add(instance);
                MajNonDanger.add(instance); // majority without mindanger
            }
        }

        // compute Value Distance Metric matrices for nominal features
        Map vdmMap = new HashMap();
        Enumeration attrEnum = getInputFormat().enumerateAttributes();
        while (attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            if (!attr.equals(getInputFormat().classAttribute())) {
                if (attr.isNominal() || attr.isString()) {
                    double[][] vdm = new double[attr.numValues()][attr.numValues()];
                    vdmMap.put(attr, vdm);
                    int[] featureValueCounts = new int[attr.numValues()];
                    int[][] featureValueCountsByClass = new int[getInputFormat().classAttribute().numValues()][attr.numValues()];
                    instanceEnum = getInputFormat().enumerateInstances();
                    while (instanceEnum.hasMoreElements()) {
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
        int extraIndicesCount = (int) (percentageRemainder * Minority.numInstances());
        if (extraIndicesCount >= 1) {
            for (int i = 0; i < Minority.numInstances(); i++) {
                extraIndices.add(i);
            }
        }
        Collections.shuffle(extraIndices, rand);
        extraIndices = extraIndices.subList(0, extraIndicesCount);
        Set extraIndexSet = new HashSet(extraIndices);

        // the main loop to handle computing nearest neighbors and generating DBHCABSMOTE
       for (int i = Majority.numInstances() - 1; i >= 0; i--)   // remove majority border from MajNonDanger
       {
           System.out.println("Majority number instance " + i);
           RemoveRange remove = new RemoveRange();
           remove.setInputFormat(MajNonDanger);
           remove.setInvertSelection(false);
           Instance instanceI = Majority.instance(i);

           // find k nearest neighbors for each instance
           List distanceToInstance = new LinkedList();
           for (int j = 0; j < total.numInstances(); j++) {
               Instance instanceJ = total.instance(j);
               if (i != j) {
                   double distance = 0;
                   attrEnum = getInputFormat().enumerateAttributes();
                   while (attrEnum.hasMoreElements()) {
                       Attribute attr = (Attribute) attrEnum.nextElement();
                       if (!attr.equals(getInputFormat().classAttribute())) {
                           double iVal = instanceI.value(attr);
                           double jVal = instanceJ.value(attr);
                           if (attr.isNumeric()) {
                               distance += Math.pow(iVal - jVal, 2);
                           } else {
                               distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
                           }
                       }
                   }
                   distance = Math.pow(distance, .5);
                   distanceToInstance.add(new Object[]{distance, instanceJ});
               }
           }

           // sort the neighbors according to distance
           Collections.sort(distanceToInstance, new Comparator() {
               public int compare(Object o1, Object o2) {
                   double distance1 = (Double) ((Object[]) o1)[0];
                   double distance2 = (Double) ((Object[]) o2)[0];
                   return Double.compare(distance1, distance2);
               }
           });

           // populate the actual nearest neighbor instance array
           Iterator entryIterator = distanceToInstance.iterator();
           int j = 0;
           int mm = 0; // m' ( number of Minority samples)

           while (entryIterator.hasNext() && j < nearestNeighbors) {
               Instance inst = (Instance) ((Object[]) entryIterator.next())[1];
               int clsI = (int) inst.classValue();
               if (clsI == minIndex)
                   mm++;
               j++;
           }

           // remove of Majority borderline instance , and create MajNonDanger which consist of majority instances without maj borderline instance
           if (mm >= (nearestNeighbors / 2) && mm < nearestNeighbors) {
               remove.setInstancesIndices("" + (i + 1));
               MajNonDanger = Filter.useFilter(MajNonDanger, remove);
           }
       }

        System.out.println("\n");
        for (int i = 0; i < Minority.numInstances(); i++) {
            if ((i + 1) % 100 == 1) {
                System.out.println("Minority number instance " + (i + 1) + "/" + Minority.numInstances());
            }
            Instance instanceI = Minority.instance(i);

            // find k nearest neighbors for each instance
            List distanceToInstance = new LinkedList();
            for (int j = 0; j < total.numInstances(); j++) {
                Instance instanceJ = total.instance(j);
                if (i != j) {
                    double distance = 0;
                    attrEnum = getInputFormat().enumerateAttributes();
                    while (attrEnum.hasMoreElements()) {
                        Attribute attr = (Attribute) attrEnum.nextElement();
                        if (!attr.equals(getInputFormat().classAttribute())) {
                            double iVal = instanceI.value(attr);
                            double jVal = instanceJ.value(attr);
                            if (attr.isNumeric()) {
                                distance += Math.pow(iVal - jVal, 2);
                            } else {
                                distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
                            }
                        }
                    }
                    distance = Math.pow(distance, .5);
                    distanceToInstance.add(new Object[]{distance, instanceJ});
                }
            }

            // sort the neighbors according to distance
            Collections.sort(distanceToInstance, new Comparator() {
                public int compare(Object o1, Object o2) {
                    double distance1 = (Double) ((Object[]) o1)[0];
                    double distance2 = (Double) ((Object[]) o2)[0];
                    return Double.compare(distance1, distance2);
                }
            });

            // populate the actual nearest neighbor instance array
            Iterator entryIterator = distanceToInstance.iterator();
            int j = 0;
            int mm = 0; // m' ( number of majority samples)

            while (entryIterator.hasNext() && j < nearestNeighbors) {
                Instance inst = (Instance) ((Object[]) entryIterator.next())[1];
                int clsI = (int) inst.classValue();
                if (clsI != minIndex)
                    mm++;
                j++;
            }

            // Creation of Minority mindanger set, and Minority non mindanger set
            if (mm >= (nearestNeighbors / 2) && mm < nearestNeighbors) {
                mindanger.add(instanceI);
            } else
                MinNonDanger.add(instanceI);
        }

        System.out.println("\n\nNumber of mindanger instances: " + mindanger.size());

        // Apply clustering on the Minority mindanger set
        int tempclassIndex = mindanger.classIndex();
        mindanger.setClassIndex(-1);  //  unset the class before you pass the data to DBSCAN.

        // Passing the parameters needed to DBSCAN
        DBSCAN dbscan = new DBSCAN();
        dbscan.setEpsilon(getEpsilon());
        dbscan.setMinPoints(getMinPts());

        // Executing DBSCAN on mindanger instances
        dbscan.buildClusterer(mindanger);

        System.out.println("\n\nNumber of clusters created after using DBSCAN: " + dbscan.numberOfClusters());

        weka.core.Instances[] dataset = new weka.core.Instances[dbscan.numberOfClusters()];
        for (int i = 0; i < dataset.length; i++) {
            dataset[i] = new Instances(mindanger, 0);
        }
        var replaceMissingValues_Filter = new ReplaceMissingValues();
        replaceMissingValues_Filter.setInputFormat(mindanger);
        Instances filteredInstances = Filter.useFilter(mindanger, replaceMissingValues_Filter);

        for (int i = 0; i < mindanger.size(); i++) {
            instanceMapping.put(mindanger.instance(i), filteredInstances.instance(i));
        }

        // Map the instances in instanceDataObjectMapping to store its memory
        database = new Database(dbscan.getDistanceFunction(), mindanger);
        for (int i = 0; i < database.getInstances().numInstances(); i++) {
            DataObject dataObject = new DataObject(
                    database.getInstances().instance(i),
                    Integer.toString(i),
                    database);
            database.insert(dataObject);
            instanceDataObjectMapping.put(filteredInstances.instance(i), dbscan.database.getDataObject(Integer.toString(i)));
        }

        for (Instance inst : mindanger) {
            try {
                int clusterLabel = clusterInstance(inst);
                if (clusterLabel != -2) {
                    dataset[clusterLabel].add(inst);
                }
            } catch (Exception e) {
                System.out.println(Arrays.toString(e.getStackTrace()));
            }
        }

        for (int i = 0; i < dbscan.numberOfClusters(); i++) {
            System.out.println("Cluster Size: " + i + ", " + dataset[i].size());
        }

        mindanger.setClassIndex(tempclassIndex);
        for (int i = 0; i < dbscan.numberOfClusters(); i++) {
            dataset[i].setClassIndex(tempclassIndex);
        }

        Double per[] = new Double[dbscan.numberOfClusters()];

        for (int i = 0; i < dbscan.numberOfClusters(); i++) {
            per[i] = (double) dataset[i].numInstances() / (double) mindanger.numInstances();
        }

        int permaxValueindex = 0;
        double permaxValue = per[0];
        for (int i = 0; i < per.length; i++) {
            if (per[i] > permaxValue) {
                permaxValue = per[i];
                permaxValueindex = i;
            }
        }

        System.out.println("\n\nLargest cluster is cluster[" + permaxValueindex + "] = " + (permaxValue));

        int perminValueindex = 0;
        double perminValue = per[0];
        for (int i = 0; i < per.length; i++) {
            if (per[i] < perminValue) {
                perminValue = per[i];
                perminValueindex = i;
            }
        }

        Double perdatatomax[] = new Double[dbscan.numberOfClusters()];
        for (int i = 0; i < dbscan.numberOfClusters(); i++) {
            perdatatomax[i] = (double) dataset[i].numInstances() / (double) dataset[permaxValueindex].numInstances();
        }

        // Generate synthetic instances based on K nearest neighbors
        Instance[] nnArray = new Instance[knearestNeighbors];

        for (int a = 0; a < per.length; a++) {
            int n = dataset[permaxValueindex].numInstances();
            int l = dataset[a].numInstances();

            if (perdatatomax[a] >= 0.50) {
                Newtotal.addAll(dataset[a]);

                for (int i = 0; i < dataset[a].numInstances(); i++) {
                    Instance instanceI = dataset[a].instance(i);

                    // find k nearest neighbors for each instance
                    List distanceToInstance = new LinkedList();
                    for (int j = 0; j < dataset[a].numInstances(); j++) {
                        Instance instanceJ = dataset[a].instance(j);
                        if (i != j) {
                            double distance = 0;
                            attrEnum = getInputFormat().enumerateAttributes();
                            while (attrEnum.hasMoreElements()) {
                                Attribute attr = (Attribute) attrEnum.nextElement();
                                if (!attr.equals(getInputFormat().classAttribute())) {
                                    double iVal = instanceI.value(attr);
                                    double jVal = instanceJ.value(attr);
                                    if (attr.isNumeric()) {
                                        distance += Math.pow(iVal - jVal, 2);
                                    } else {
                                        distance += ((double[][]) vdmMap.get(attr))[(int) iVal][(int) jVal];
                                    }
                                }
                            }
                            distance = Math.pow(distance, .5);
                            distanceToInstance.add(new Object[]{distance, instanceJ});
                        }

                    }

                    Collections.sort(distanceToInstance, new Comparator() {
                        public int compare(Object o1, Object o2) {
                            double distance1 = (Double) ((Object[]) o1)[0];
                            double distance2 = (Double) ((Object[]) o2)[0];
                            return Double.compare(distance1, distance2);
                        }
                    });

                    // populate the actual nearest neighbor instance array nnArray between mindanger (I) and Majoritysample
                    Iterator entryIterator = distanceToInstance.iterator();
                    int z = 0;
                    while (entryIterator.hasNext() && z < knearestNeighbors) {
                        nnArray[z] = (Instance) ((Object[]) entryIterator.next())[1];
                        z++;
                    }

                    int neighborCount = Math.min(knearestNeighbors, z);
                    if (neighborCount != 0) {
                        while (l < (dataset[a].numInstances() * getPercentage()) || extraIndexSet.remove(i)) {
                            double[] values = new double[Minority.numAttributes()];
                            int nn = rand.nextInt(neighborCount);
                            attrEnum = getInputFormat().enumerateAttributes();
                            while (attrEnum.hasMoreElements()) {
                                Attribute attr = (Attribute) attrEnum.nextElement();
                                if (!attr.equals(getInputFormat().classAttribute())) {
                                    if (attr.isNumeric()) {
                                        double dif = nnArray[nn].value(attr) - instanceI.value(attr);
                                        double gap = rand.nextDouble() % getGapRange();
                                        values[attr.index()] = (double) (instanceI.value(attr) + gap * dif);
                                    } else if (attr.isDate()) {
                                        double dif = nnArray[nn].value(attr) - instanceI.value(attr);
                                        double gap = rand.nextDouble() % getGapRange();
                                        values[attr.index()] = (long) (instanceI.value(attr) + gap * dif);
                                    } else {
                                        int[] valueCounts = new int[attr.numValues()];
                                        int iVal = (int) instanceI.value(attr);
                                        valueCounts[iVal]++;
                                        for (int nnEx = 0; nnEx < neighborCount; nnEx++) {
                                            int val = (int) nnArray[nnEx].value(attr);
                                            valueCounts[val]++;
                                        }
                                        int maxIndex = 0;
                                        int max = Integer.MIN_VALUE;
                                        for (int index = 0; index < attr.numValues(); index++) {
                                            if (valueCounts[index] > max) {
                                                max = valueCounts[index];
                                                maxIndex = index;
                                            }
                                        }
                                        values[attr.index()] = maxIndex;
                                    }
                                }
                            }

                            values[Minority.classIndex()] = minIndex;
                            Instance synthetic = new DenseInstance(1.0, values);
                            push(synthetic);
                            Generated.add(synthetic);
                            l++;
                        }
                    }
                }
            }
        }

        // Combining Majority + Minority Minority without mindanger area
        Newtotal.addAll(MajNonDanger); // add majority instances without majority border instances
        Newtotal.addAll(MinNonDanger); // add Minority instances which are not border instances
        Newtotal.addAll(Generated);

        int MinorityInstances = MinNonDanger.size() + Generated.size();
        System.out.println("\n\nAll Instances: " + Newtotal.size());
        System.out.println("Majority Instances: " + MajNonDanger.size());
        System.out.println("Minority Instances: " + MinorityInstances);
        System.out.println("Generated Instances: " + Generated.size());
        double classBalanceValue = Math.abs((double) MajNonDanger.size() / MinorityInstances);
        System.out.println("Dataset Class Balance Value: " + classBalanceValue);

        // exporting new dataset
        BufferedWriter writer = new BufferedWriter(new FileWriter("./data/DBHCABSMOTE.arff"));
        writer.write(Newtotal.toString());
        writer.flush();
        writer.close();
        System.out.println("The new generated dataset file is exported to into the data folder.");
    }

    /**
     * Main method for running this filter.
     *
     * @param args should contain arguments to the filter:
     *             use -h for help
     */
    public static void main(String[] args) {
        runFilter(new DBHCABSMOTE(), args);
    }
}
 
