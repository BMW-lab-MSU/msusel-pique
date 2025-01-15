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
import pique.evaluation.ProbabilityDensityFunctionUtilityFunction;


import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVSaver;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.Logistic;
import weka.core.WekaPackageManager;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import weka.core.Attribute;
import weka.core.DenseInstance;

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
    private String[] msNames;
    private BigDecimalWithContext[][] manWeights;
    private BigDecimalWithContext[][] comparisonMat;
    private BigDecimalWithContext[][] measMat;
    private int numQA;
    private int numPF;
    private int numMS;


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
                        || modelMSNames.contains(getCategoryName(data[0],msPrefix))) {
                    if (lineCount < numQA+1) { //tqi weights, fill values for ahpMat
                        for (int i = 1; i < data.length; i++) {
                            comparisonMat[lineCount-1][i-1] = new BigDecimalWithContext(Double.parseDouble(data[i].trim()));
                        }
                    }
                    else  { //QA weights, fill values for manWeights
                        //parse out the integer for the CWE number and add appropriate prefix, unless it is not numbered
                        pfNames[lineCount-numQA-1] = getCategoryName(data[0],pfPrefix);
                        for (int i = 1; i < data.length; i++) {
                            manWeights[lineCount-numQA-1][i-1] = new BigDecimalWithContext(Double.parseDouble(data[i].trim()));
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
            mlWeights(qualityModel.getQualityAspects().values(),weights);
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
     */
    private void mlWeights(Collection<ModelNode> nodes, Set<WeightResult> weights) throws Exception{
        BigDecimal[][] normMat = normalizeByColSum(manWeights);
        ProbabilityDensityFunctionUtilityFunction probabilityDensityFunctionUtilityFunction;
        probabilityDensityFunctionUtilityFunction = new ProbabilityDensityFunctionUtilityFunction();

        for (ModelNode node : nodes) {
            WeightResult weightResult = new WeightResult(node.getName());
            System.out.println(node.getName());

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

//            System.out.println(ref.data);

            // Saving the dataset
            CSVSaver saver = new CSVSaver();
            saver.setInstances(ref.data);
            saver.setFile(new File("./src/test/out/ML/"+ node.getName()+"_measures.csv"));
            saver.setDestination(new File("./src/test/out/ML/"+ node.getName()+"_measures.csv"));
            saver.writeBatch();


            try {
                LinearRegression model = testWeka(ref.data);
//                node.setThresholds(model);
                // Saving the model
                System.out.println("----------saving model---------------");
                weka.core.SerializationHelper.write("./src/test/out/ML/"+node.getName()+"_lin.model",model);


            } catch (Exception e) {
                e.printStackTrace();
            }



//            weights.add(weightResult);
        }
    }

    private static BigDecimal[] addToArray(BigDecimal[] array, BigDecimal newElement) {
        BigDecimal[] newArray = Arrays.copyOf(array, array.length + 1);
        newArray[newArray.length - 1] = newElement;
        return newArray;
    }

    public static LinearRegression testWeka(Instances dataset) throws Exception{
        //Load Data set

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
        LinearRegression model = new LinearRegression();
//        Logistic model = new Logistic();
        model.buildClassifier(train);
        //output model
        System.out.println("LR FORMULA : "+model);

//        // Saving the model
//        System.out.println("----------saving model---------------");
//        weka.core.SerializationHelper.write("./lin.model",model);
//
//        // loading the model
//        System.out.println("----------loading model---------------");
//        Classifier loaded_model = (Classifier) weka.core.SerializationHelper.read("./lin.model");
//
//        // Now Predicting the cost
//        Instance myHouse = test.lastInstance();
//        double price = loaded_model.classifyInstance(myHouse);
//        System.out.println("-------------------------");
//        System.out.println(myHouse);
//        System.out.println("PREDICTING THE score : "+price);



        return model;
    }

//    public static void attTest() throws Exception {
//        ArrayList<Attribute> atts;
//        ArrayList<Attribute> attsRel;
//        ArrayList<String>    attVals;
//        ArrayList<String>    attValsRel;
//        Instances            data;
//        Instances            dataRel;
//        double[]             vals;
//        double[]             valsRel;
//        int                  i;
//
//        // 1. set up attributes
//        atts = new ArrayList<Attribute>();
//        // - numeric
//        atts.add(new Attribute("att1"));
//        // - nominal
//        attVals = new ArrayList<String>();
//        for (i = 0; i < 5; i++)
//            attVals.add("val" + (i+1));
//        atts.add(new Attribute("att2", attVals));
//        // - string
//        atts.add(new Attribute("att3", (ArrayList<String>) null));
//        // - date
//        atts.add(new Attribute("att4", "yyyy-MM-dd"));
//        // - relational
//        attsRel = new ArrayList<Attribute>();
//        // -- numeric
//        attsRel.add(new Attribute("att5.1"));
//        // -- nominal
//        attValsRel = new ArrayList<String>();
//        for (i = 0; i < 5; i++)
//            attValsRel.add("val5." + (i+1));
//        attsRel.add(new Attribute("att5.2", attValsRel));
//        dataRel = new Instances("att5", attsRel, 0);
//        atts.add(new Attribute("att5", dataRel, 0));
//
//        // 2. create Instances object
//        data = new Instances("MyRelation", atts, 0);
//
//        // 3. fill with data
//        // first instance
//        vals = new double[data.numAttributes()];
//        // - numeric
//        vals[0] = Math.PI;
//        // - nominal
//        vals[1] = attVals.indexOf("val3");
//        // - string
//        vals[2] = data.attribute(2).addStringValue("This is a string!");
//        // - date
//        vals[3] = data.attribute(3).parseDate("2001-11-09");
//        // - relational
//        dataRel = new Instances(data.attribute(4).relation(), 0);
//        // -- first instance
//        valsRel = new double[2];
//        valsRel[0] = Math.PI + 1;
//        valsRel[1] = attValsRel.indexOf("val5.3");
//        dataRel.add(new DenseInstance(1.0, valsRel));
//        // -- second instance
//        valsRel = new double[2];
//        valsRel[0] = Math.PI + 2;
//        valsRel[1] = attValsRel.indexOf("val5.2");
//        dataRel.add(new DenseInstance(1.0, valsRel));
//        vals[4] = data.attribute(4).addRelation(dataRel);
//        // add
//        data.add(new DenseInstance(1.0, vals));
//
//        // second instance
//        vals = new double[data.numAttributes()];  // important: needs NEW array!
//        // - numeric
//        vals[0] = Math.E;
//        // - nominal
//        vals[1] = attVals.indexOf("val1");
//        // - string
//        vals[2] = data.attribute(2).addStringValue("And another one!");
//        // - date
//        vals[3] = data.attribute(3).parseDate("2000-12-01");
//        // - relational
//        dataRel = new Instances(data.attribute(4).relation(), 0);
//        // -- first instance
//        valsRel = new double[2];
//        valsRel[0] = Math.E + 1;
//        valsRel[1] = attValsRel.indexOf("val5.4");
//        dataRel.add(new DenseInstance(1.0, valsRel));
//        // -- second instance
//        valsRel = new double[2];
//        valsRel[0] = Math.E + 2;
//        valsRel[1] = attValsRel.indexOf("val5.1");
//        dataRel.add(new DenseInstance(1.0, valsRel));
//        vals[4] = data.attribute(4).addRelation(dataRel);
//        // add
//        data.add(new DenseInstance(1.0, vals));
//
//        // 4. output data
//        System.out.println(data);
//    }

}

