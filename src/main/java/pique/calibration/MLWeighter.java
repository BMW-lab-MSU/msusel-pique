/**
 * MIT License
 * Copyright (c) 2019 Montana State University Software Engineering Labs
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/// Copied from pique-bin-docker

package pique.calibration;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.util.*;

import org.apache.commons.lang3.ArrayUtils;

import pique.calibration.IWeighter;
import pique.calibration.WeightResult;
import pique.model.ModelNode;
import pique.model.QualityModel;
import pique.utility.BigDecimalWithContext;
import pique.utility.PiqueProperties;
import pique.evaluation.ProbabilityDensityFunctionUtilityFunction;


import weka.core.*;
import weka.classifiers.Classifier;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVSaver;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.util.ArrayList;

import static java.lang.Long.sum;

/**
 * @author Andrew Johnson
 *
 * This class should weight based off of pairwise comparisons for the quality aspect level but utilize manual weights for
 * product factors to the quality aspect level. This allows stakeholder
 * interest to be represented but reduces the extreme amount of comparisons.
 */
public class MLWeighter implements IWeighter{
    private String[] qaNames;
    private String[] pfNames;
    public String[] msNames;
    private BigDecimalWithContext[][] manWeights;
    private BigDecimalWithContext[][] comparisonMat;
    private BigDecimalWithContext[][] measMat;
    private int numQA;
    private int numPF;
    public int numMS;


    /**
     * This class will take a quality model as input and will create the weighting for edges from each layer to the next.
     * This weighter uses averages to weight edges into each node up to the product factor level, then uses manual weighting
     * as specified by the .csv comparisons for Product Factor to Quality Aspects, then uses that same file to find comparisons
     * for weighting the Quality Aspects to the TQI using the AHP.
     *
     * @param qualityModel The QualityModel to instantiate weights for
     * @param externalInput Used to pass in path to comparison matrix in the case of non-standard location, for example when testing
     */
    @Override
    public Set<WeightResult> elicitateWeights(QualityModel qualityModel, Path... externalInput) {
        Properties prop = PiqueProperties.getProperties();

        numQA = qualityModel.getQualityAspects().size();
        numPF = qualityModel.getProductFactors().size();
        numMS = qualityModel.getMeasures().size();

        qaNames = new String[numQA];
        pfNames = new String[numPF];
        msNames = new String[numMS];

        manWeights = new BigDecimalWithContext[numPF][numQA];
        comparisonMat = new BigDecimalWithContext[numQA][numQA];
        measMat = new BigDecimalWithContext[numMS][numQA];

        List<String> modelQANames = new ArrayList<String>();
        List<String> modelPFNames = new ArrayList<String>();
        List<String> modelMSNames = new ArrayList<String>();

        qualityModel.getQualityAspects().values().stream().forEach(s -> modelQANames.add(s.getName()));
        qualityModel.getProductFactors().values().stream().forEach(s -> modelPFNames.add(s.getName()));
        qualityModel.getMeasures().values().stream().forEach(s -> modelMSNames.add(s.getName()));



        String pathToCsv = "src/main/resources/comparisons.csv";
        if (externalInput.length>0) pathToCsv = externalInput[0].toString();

        String pfPrefix = "Category ";
        String msPrefix = "Measure ";
        BufferedReader csvReader;
        int lineCount = 0;
        try {
            csvReader = new BufferedReader(new FileReader(pathToCsv));

            String row;
            while (( row = csvReader.readLine()) != null) {
                String[] data = row.split(",");
                //first line is qaNames
                if (lineCount == 0) {
                    for (int i = 1; i < data.length; i++) {
                        qaNames[i-1] = data[i];
                    }
                    lineCount++;
                }
                //otherwise, check if the first entry of data is part of the model
                else if (modelQANames.contains(data[0]) || modelPFNames.contains(getCategoryName(data[0],pfPrefix))
                        || modelMSNames.contains(data[0])) {
                    if (lineCount < numQA+1) { //tqi weights, fill values for ahpMat
                        for (int i = 1; i < data.length; i++) {
                            comparisonMat[lineCount-1][i-1] = new BigDecimalWithContext(Double.parseDouble(data[i].trim()));
                        }
                    }
                    else if (lineCount < numQA+numPF+1)  { //QA weights, fill values for manWeights
                        //parse out the integer for the CWE number and add appropriate prefix, unless it is not numbered
                        pfNames[lineCount-numQA-1] = getCategoryName(data[0],pfPrefix);
                        for (int i = 1; i < data.length; i++) {
                            manWeights[lineCount - numQA - 1][i - 1] = new BigDecimalWithContext(Double.parseDouble(data[i].trim()));
                        }
                    } else {
                        msNames[lineCount-numQA-numPF-1] = data[0];
                        for (int i = 1; i < data.length; i++) {
                            measMat[lineCount - numQA - numPF - 1][i - 1] = new BigDecimalWithContext(Double.parseDouble(data[i].trim()));
                        }
                    }
                    lineCount++;
                }
            }
            csvReader.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }


        Set<WeightResult> weights = new HashSet<>();

        //there is certainly a better way to implement this next part

        //set the weights as a simple 1/number of children for each edge
        averageWeights(qualityModel.getMeasures().values(),weights);
        averageWeights(qualityModel.getDiagnostics().values(),weights);
        averageWeights(qualityModel.getProductFactors().values(),weights);


        //set the weights for edges going into quality aspects based on manual weighting
        manualWeights(qualityModel.getQualityAspects().values(),weights);

        //set the weights for edges using ML
        try {
            mlWeights(qualityModel.getQualityAspects().values(),weights, measMat);
        } catch (Exception e) {
            e.printStackTrace();
        }

//        try {
//            testWeka();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

//        try {
//            attTest();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }


        //set the weights for edges going into tqi based on ahp
        ahpWeights(qualityModel.getTqi(), weights);

        return weights;
    }






    @Override
    public String getName() {
        return this.getClass().getCanonicalName();
    }

    /**
     * This class will take a name and prefix and add the prefix to the name, as well as "CWE-". This is to convert the node name from the comparison
     * file to the specific strings that are used in the quality model as of right now. This could easily break if the file or quality model changes.
     * @param name The name of the model node as given in the comparison file.
     * @param pfPrefix The common prefix for all categories
     * @return The Category in the model based upon the given name.
     */
    private String getCategoryName(String name, String pfPrefix) {
        if (name.replaceAll("[\\D]", "").length() == 0) {
            //no numbers in the name
            return pfPrefix +name;
        }
        else {
            return pfPrefix + "CWE-" + Integer.toString(Integer.parseInt(name.replaceAll("[\\D]", "")));
        }
    }

    /**
     * This class will take a name and prefix and add the prefix to the name, as well as "CWE-". This is to convert the node name from the comparison
     * file to the specific strings that are used in the quality model as of right now. This could easily break if the file or quality model changes.
     * @param name The name of the model node as given in the comparison file.
     * @param msPrefix The common prefix for all categories
     * @return The Category in the model based upon the given name.
     */
    private String getMeasureName(String name, String msPrefix) {
        if (name.replaceAll("[\\D]", "").length() == 0) {
            //no numbers in the name
            return msPrefix +name;
        }
        else {
            return msPrefix + "CWE-" + Integer.toString(Integer.parseInt(name.replaceAll("[\\D]", "")));
        }
    }


    /**
     * Sets the incoming weights for a node to evaluate the average value of children.
     * @param values Nodes to set weights for
     * @param weights A set to keep track of weights
     */
    private void averageWeights(Collection<ModelNode> values, Set<WeightResult> weights) {
        values.forEach(node -> {
            WeightResult weightResult = new WeightResult(node.getName());
            node.getChildren().values().forEach(child -> weightResult.setWeight(child.getName(), averageWeight(node)));
            weights.add(weightResult);
        });
    }

    /**
     * Calculates 1/(number of children) for the current node.
     * @param currentNode
     * @return 1/currentNode.numberOfChildren
     */
    private BigDecimal averageWeight(ModelNode currentNode) {
        return new BigDecimalWithContext(1.0).divide(
                new BigDecimalWithContext(currentNode.getChildren().size()),
                BigDecimalWithContext.getMC());
    }


    /**
     * This sets the incoming weights for a node based on AHP.
     * @param tqi The TQI node of the model, or the node for which to calculate the incoming edges.
     * @param weights A set to keep track of weights.
     */
    private void ahpWeights(ModelNode tqi, Set<WeightResult> weights) {
        BigDecimal[] ahpVec = new BigDecimal[numQA];
        //normalize by column sums
        BigDecimal[][] norm = normalizeByColSum(comparisonMat);
        //get the row means
        for (int i= 0; i < numQA; i++) {
            ahpVec[i] = rowMean(norm,i);
        }
        WeightResult weightResult = new WeightResult(tqi.getName());
        tqi.getChildren().values().forEach(child ->
                weightResult.setWeight(child.getName(), ahpVec[ArrayUtils.indexOf(qaNames, child.getName())]));
        weights.add(weightResult);
    }

    /**
     * Normalize a 2d array by column sum such that every value is that value divided by the sum of the column.
     */
    private BigDecimal[][] normalizeByColSum(BigDecimal[][] mat) {
        BigDecimal[][] norm = new BigDecimal[mat.length][mat[0].length];
        for (int i= 0; i < mat.length; i++) {
            for (int j= 0; j < mat[0].length; j++) {
                norm[i][j] = (mat[i][j].divide(colSum(mat,j),BigDecimalWithContext.getMC()));
            }
        }
        return norm;
    }

    /**
     * Normalize a 2d array by row sum such that every value is that value divided by the sum of the row.
     */
    private BigDecimal[][] normalizeByRowSum(BigDecimal[][] mat) {
        BigDecimal[][] norm = new BigDecimal[mat.length][mat[0].length];
        for (int i= 0; i < mat.length; i++) {
            for (int j= 0; j < mat[0].length; j++) {
                norm[i][j] = (mat[i][j].divide(rowSum(mat,i),BigDecimalWithContext.getMC()));
            }
        }
        return norm;
    }

    private BigDecimal rowMean(BigDecimal[][] mat, int row) {
        int cols = mat[0].length;
        return (rowSum(mat,row)).divide(new BigDecimalWithContext(cols), BigDecimalWithContext.getMC());
    }

    private BigDecimal rowSum(BigDecimal[][] mat, int row) {
        int cols = mat[0].length;
        BigDecimal sumRow = new BigDecimalWithContext(0);
        for(int j = 0; j < cols; j++){
            sumRow = sumRow.add(mat[row][j]);
        }
        return sumRow;
    }

    private BigDecimal colSum(BigDecimal[][] mat, int col) {
        int rows = mat.length;
        BigDecimal sumCol = new BigDecimalWithContext(0);
        for(int j = 0; j < rows; j++){
            sumCol = sumCol.add(mat[j][col]);
        }
        return (sumCol);
    }

    /***
     * Add an element to a BigDecimal array
     * @param array
     * @param newElement
     * @return
     */
    private static BigDecimal[] addToArray(BigDecimal[] array, BigDecimal newElement) {
        BigDecimal[] newArray = Arrays.copyOf(array, array.length + 1);
        newArray[newArray.length - 1] = newElement;
        return newArray;
    }


    /**
     * Weight edges based on manual decisions from the comparisons
     * @param nodes nodes to set weights for
     * @param weights Set to keep track of weights
     */
    private void manualWeights(Collection<ModelNode> nodes, Set<WeightResult> weights) {
        BigDecimal[][] normMat = normalizeByColSum(manWeights);


        for (ModelNode node : nodes) {
            WeightResult weightResult = new WeightResult(node.getName());
            node.getChildren().values().forEach(child ->
                    weightResult.setWeight(child.getName(), normMat[ArrayUtils.indexOf(pfNames, child.getName())][ArrayUtils.indexOf(qaNames, node.getName())]));
            weights.add(weightResult);
        }
    }


    /**
     * Weight edges based on ml decisions from the comparisons
     * @param nodes nodes to set weights for
     * @param weights Set to keep track of weights
     * @param measMat
     */
    private void mlWeights(Collection<ModelNode> nodes, Set<WeightResult> weights, BigDecimal[][] measMat) throws Exception{

        //Data source priority
        // TODO: read this parameter from properties
//        String priority = "CSV";
        String priority = "IndirectChildren";

        for (ModelNode node : nodes) {
            WeightResult weightResult = new WeightResult(node.getName());
            System.out.println(node.getName());

            // Creating Dataset
            // https://waikato.github.io/weka-wiki/formats_and_processing/creating_arff_file/

            Instances data = null;
            try{
                if (priority.equals("CSV")) {
                    data = dataFromCsv(node, measMat);
                }else if (priority.equals("Children")) {
                    data = dataFromStructure(node);
                }else if (priority.equals("IndirectChildren")) {
                    data = dataFromIndirectStructure(node);
                }else {
                    System.out.println("Unknown priority source :" + priority);
                }

            } catch (Exception e){
                e.printStackTrace();
            }




//            System.out.println(data);

            // Saving the dataset
            CSVSaver saver = new CSVSaver();
            saver.setInstances(data);
            saver.setFile(new File("./src/test/out/ML/"+ node.getName()+"_measures.csv"));
            saver.setDestination(new File("./src/test/out/ML/"+ node.getName()+"_measures.csv"));
            saver.writeBatch();


            try {
                assert data != null;
                Classifier model = testWeka(data);
                // Saving the model
                System.out.println("----------saving model---------------");
                SerializationHelper.write("./src/test/out/ML/"+node.getName()+"_lin.model",model);
            } catch (Exception e) {
                e.printStackTrace();
            }

//            weights.add(weightResult);
        }
    }


    public Instances dataFromCsv(ModelNode node, BigDecimal[][] measMat) throws Exception{
        ProbabilityDensityFunctionUtilityFunction probabilityDensityFunctionUtilityFunction;
        probabilityDensityFunctionUtilityFunction = new ProbabilityDensityFunctionUtilityFunction();

        var ref = new Object() {
            int n_measures = 0;
            int n_projects = 0;

            ArrayList<Attribute> atts;
            Instances            data;
            double[]             vals;
            ArrayList<BigDecimal[]> arrayVals;
        };

        // 1. set up attributes
        ref.atts = new ArrayList<Attribute>();
        ref.arrayVals = new ArrayList<>();

        List<String> msNameList = Arrays.asList(this.msNames);
        List<String> qaNameList = Arrays.asList(this.qaNames);
        int qaIndex = qaNameList.indexOf(node.getName());

        for(int i =0; i< this.numMS; i++ ){
            System.out.println(msNameList.get(i));
            System.out.println(measMat[i][qaIndex]);
            boolean a = (measMat[i][qaIndex] != null);
            boolean b = (measMat[i][qaIndex].compareTo(new BigDecimalWithContext(0.0)) !=0);

            if (a && b) {
                ref.atts.add(new Attribute(msNameList.get(i)));
            }
        }

        ref.atts.add(new Attribute("Average"));
        ref.data = new Instances(node.getName(), ref.atts, 0);

        System.out.println(ref.data);
        return ref.data;
    }


    public static Instances dataFromStructure(ModelNode node) throws Exception {
        ProbabilityDensityFunctionUtilityFunction probabilityDensityFunctionUtilityFunction;
        probabilityDensityFunctionUtilityFunction = new ProbabilityDensityFunctionUtilityFunction();

        var ref = new Object() {
            int n_measures = 0;
            int n_projects = 0;

            ArrayList<Attribute> atts;
            Instances            data;
            double[]             vals;
            ArrayList<BigDecimal[]> arrayVals;
        };

        // 1. set up attributes
        ref.atts = new ArrayList<Attribute>();
        ref.arrayVals = new ArrayList<>();

        node.getChildren().values().forEach(child ->{
                    System.out.println(child.getName());
//                    weightResult.setWeight(child.getName(), normMat[ArrayUtils.indexOf(pfNames, child.getName())][ArrayUtils.indexOf(qaNames, node.getName())]);


                    child.getChildren().values().forEach( grandChild ->{
                        System.out.println(grandChild.getName());

                        if (!ref.atts.contains(new Attribute(grandChild.getName()))) {
                            ref.n_measures += 1;
                            ref.atts.add(new Attribute(grandChild.getName()));

                            BigDecimal[] thresholds = grandChild.getThresholds();

                            ref.n_projects = thresholds.length;

                            BigDecimal[] scores = new BigDecimal[0];

                            // Generating PDF scores for the thresholds
                            for(int i=0; i<ref.n_projects; i++){
                                BigDecimal inValue = thresholds[i];
                                BigDecimal score = probabilityDensityFunctionUtilityFunction.utilityFunction(inValue, thresholds, false);
                                scores = addToArray(scores, score);
                            }

                            ref.arrayVals.add(scores);

                        } else {
                            System.out.println("***** double present");
                        }
                    });
                }
        );


        // FIXME: Have to do average separately outside of this loop
        ref.atts.add(new Attribute("Average"));
        ref.data = new Instances(node.getName(), ref.atts, 0);


        for(int i =0; i<ref.n_projects; i++){
            ref.vals = new double[ref.n_measures + 1];
            for (int j = 0; j < ref.n_measures; j++) {
                ref.vals[j] = ref.arrayVals.get(j)[i].doubleValue();
            }
            int nMeasures = ref.n_measures;
            ref.vals[ref.n_measures] = Arrays.stream(Arrays.copyOfRange(ref.vals, 0, ref.n_measures)).sum()/ ref.n_measures;
            ref.data.add(new DenseInstance(1.0, ref.vals));
        }

        return ref.data;
    }


    public static Instances dataFromIndirectStructure(ModelNode node) throws Exception {
        ProbabilityDensityFunctionUtilityFunction probabilityDensityFunctionUtilityFunction;
        probabilityDensityFunctionUtilityFunction = new ProbabilityDensityFunctionUtilityFunction();

        var ref = new Object() {
            int n_measures = 0;
            int n_projects = 0;

            ArrayList<Attribute> atts;
            Instances            data;
            double[]             vals;
            ArrayList<BigDecimal[]> arrayVals;
        };

        // 1. set up attributes
        ref.atts = new ArrayList<Attribute>();
        ref.arrayVals = new ArrayList<>();


        node.getIndirectChildren().values().forEach(child ->{
            System.out.println(child.getName());

            if (!ref.atts.contains(new Attribute(child.getName()))) {
                ref.n_measures += 1;
                ref.atts.add(new Attribute(child.getName()));

                BigDecimal[] thresholds = child.getThresholds();

                ref.n_projects = thresholds.length;

                BigDecimal[] scores = new BigDecimal[0];

                // Generating PDF scores for the thresholds
                for(int i=0; i<ref.n_projects; i++){
                    BigDecimal inValue = thresholds[i];
                    BigDecimal score = probabilityDensityFunctionUtilityFunction.utilityFunction(inValue, thresholds, false);
                    scores = addToArray(scores, score);
                }

                ref.arrayVals.add(scores);

            } else {
                System.out.println("***** double present");
            }

        });

        ref.atts.add(new Attribute("Average"));
        ref.data = new Instances(node.getName(), ref.atts, 0);


        for(int i =0; i<ref.n_projects; i++){
            ref.vals = new double[ref.n_measures + 1];
            for (int j = 0; j < ref.n_measures; j++) {
                ref.vals[j] = ref.arrayVals.get(j)[i].doubleValue();
            }
            int nMeasures = ref.n_measures;
            ref.vals[ref.n_measures] = Arrays.stream(Arrays.copyOfRange(ref.vals, 0, ref.n_measures)).sum()/ ref.n_measures;
            ref.data.add(new DenseInstance(1.0, ref.vals));
        }


        return ref.data;
    }




    /**
     * Makes the predicting values to monotonically increase in each cardinal direction
     * @param inValues
     * @param model
     * @return
     */
    public static double postProcessInstance(Instance inValues, Classifier model) throws Exception {
        int n_attributes = inValues.numAttributes();
        double newScore = 0.0;

        newScore = model.classifyInstance(inValues);

        for (int i = 0; i < n_attributes; i++) {
            double att_value = inValues.value(i);
            Instance newInValues = inValues;
            for (double j = 0; j <= att_value; j=j+0.1) {
                newInValues.setValue(i,j);
                newScore = Math.max(newScore, model.classifyInstance(newInValues));

            }
        }
        return newScore;
    }

    /**
     * Removes constatnt columns from  data
     * @param data
     * @return
     *
     */

    public static Instances removeConstantAttributes(Instances data){
        Instances newData = new Instances(data);

        int numAttributes = newData.numAttributes();
        List<Attribute> atts = new ArrayList<>();

        for (int i = 0; i < numAttributes; i++) {
            Attribute att_name = data.attribute(i);
            System.out.println(att_name);

            int num_unique = data.numDistinctValues(att_name);

            if (num_unique <=1) {
                newData.deleteAttributeAt(i);
            }

        }


        return newData;
    }

    public static Classifier testWeka(Instances dataset) throws Exception{
        //Load Data set

        System.out.println("Cleaned data");
        Instances cleanedData = new Instances(removeConstantAttributes(dataset));
//        System.out.println(cleanedData);
        System.out.println(cleanedData.size());


//        DataSource source = new DataSource("/PWD/weka-3-8-6/data/regression-datasets/regression-datasets/housing.arff");
//        Instances dataset = source.getDataSet();
        //set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes()-1);


        // Train test divide

//        RemovePercentage filter = new RemovePercentage();
//
//        String[] options = new  String[4];
//
//        options[0] = "-P";
//        options[1] = "70.0F";
//        options[2] = "-V";
//        options[3] = "1";
//
//
//        filter.setOptions(options);
//        Instances train = Filter.useFilter(dataset, filter);
//
//        options[3] = "0";
//        filter.setOptions(options);
//        Instances test = Filter.useFilter(dataset, filter);


// use StratifiedRemoveFolds to randomly split the data
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

// set options for creating the subset of data
        String[] options = new String[6];

        options[0] = "-N";                 // indicate we want to set the number of folds
        options[1] = Integer.toString(5);  // split the data into five random folds
        options[2] = "-F";                 // indicate we want to select a specific fold
        options[3] = Integer.toString(1);  // select the first fold
        options[4] = "-S";                 // indicate we want to set the random seed
        options[5] = Integer.toString(1);  // set the random seed to 1

        filter.setOptions(options);        // set the filter options
        filter.setInputFormat(dataset);       // prepare the filter for the data format
        filter.setInvertSelection(false);  // do not invert the selection

// apply filter for test data here
        Instances test = Filter.useFilter(dataset, filter);
        System.out.println("test");
        System.out.println(test.size());

//  prepare and apply filter for training data here
        filter.setInvertSelection(true);     // invert the selection to get other data
        Instances train = Filter.useFilter(dataset, filter);
        System.out.println("train");
        System.out.println(train.size());

        //Build model
//        SimpleLinearRegression model = new SimpleLinearRegression();
//        LinearRegression model = new LinearRegression();
//        MultilayerPerceptron model = new MultilayerPerceptron();
        RandomForest model= new RandomForest();

        model.buildClassifier(train);
        //output model
        System.out.println("LR FORMULA : "+model);

        // Saving the model
        System.out.println("----------saving model---------------");
        weka.core.SerializationHelper.write("./lin.model",model);

        // loading the model
        System.out.println("----------loading model---------------");
        Classifier loaded_model = (Classifier) weka.core.SerializationHelper.read("./lin.model");

        // Now Predicting the cost
        Instance sampleInstance = test.lastInstance();
        double price = loaded_model.classifyInstance(sampleInstance);
        System.out.println("-------------------------");
        System.out.println(sampleInstance);
        System.out.println("PREDICTING THE SCORE : "+price);

        System.out.println("---------Post Processing----------------");
        double newScore =  0;
        newScore = postProcessInstance(sampleInstance, loaded_model);
        System.out.println("Post processing" +  newScore);

        return (Classifier) model;
    }

}

