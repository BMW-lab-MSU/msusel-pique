package pique.evaluation;

import java.lang.reflect.Array;
import java.math.BigDecimal;

import pique.evaluation.Evaluator;
import pique.model.ModelNode;
import pique.utility.BigDecimalWithContext;
import weka.classifiers.Classifier;


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
import java.util.Arrays;

import static pique.calibration.MLWeighter.addToArray;

//TODO (1.0): Documentation
public class MLEvaluator extends Evaluator {

    @Override
    public BigDecimal evaluate(ModelNode inNode) {

        // TODO (1.0): Some redesign needed to better handle quality model description where there are not yet weights,
        //  values, etc...

        String nodeName = inNode.getName();
        BigDecimal outValue = new BigDecimalWithContext("0.0");

//        BigDecimal[] inValues = new BigDecimal[0];

        var ref = new Object() {
            int n_child = 0;
            int cnt_child = 0;
            int n_projects = 0;

            ArrayList<Attribute> atts;
            Instances            data;
            double[]             vals;
            ArrayList<BigDecimal[]> arrayVals;
        };

        // 1. set up attributes
        ref.atts = new ArrayList<Attribute>();
        ref.arrayVals = new ArrayList<>();


        if (inNode.getIndirectChildren() != null) {
            System.out.println("Indirect children");
            ref.n_child = inNode.getNumIndirectChildren();
            ref.vals = new double[ref.n_child];
            inNode.getIndirectChildren().forEach((childName, child)->{
                System.out.println(child.getName());

                if (!ref.atts.contains(new Attribute(childName))) {
                    ref.atts.add(new Attribute(childName));
                    BigDecimal value = child.getValue();
                    System.out.println(value);

                    ref.vals[ref.cnt_child] = value.doubleValue();

                    ref.cnt_child += 1;
                }else{
                    System.out.println("Duplicate found");
                }
            });
        } else {
            System.out.println("Direct children");
            ref.n_child = inNode.getNumChildren();
            ref.vals = new double[ref.n_child];
            inNode.getChildren().forEach((childName, child)-> {
                System.out.println(child.getName());

                if (!ref.atts.contains(new Attribute(childName))) {
                    ref.atts.add(new Attribute(childName));
                    BigDecimal value = child.getValue();
                    System.out.println(value);

                    ref.vals[ref.cnt_child] = value.doubleValue();

                    ref.cnt_child += 1;
                }else{
                    System.out.println("Duplicate found");
                }
            });

        }

        ref.atts.add(new Attribute("Average"));
        ref.data = new Instances(inNode.getName(), ref.atts, 0);

        ref.data.add(new DenseInstance(1.0, ref.vals));
        System.out.println(ref.data);

        if (ref.data.classIndex() == -1) {
//            System.out.println("reset index...");
            ref.data.setClassIndex(ref.n_child - 1);
        }
//        ref.data.setValue(1, );

        // loading the model
        System.out.println("----------loading model---------------");
        try {
            Classifier loaded_model = (Classifier) weka.core.SerializationHelper.read("./src/test/out/ML/"+nodeName+"_lin.model");
            System.out.println(loaded_model);

            outValue = BigDecimal.valueOf(loaded_model.classifyInstance(ref.data.get(0)));

            System.out.println(outValue);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

//        // Apply weighted sums
//        for (ModelNode child : inNode.getChildren().values()) {
//            outValue = outValue.add(child.getValue().multiply(inNode.getWeight(child.getName()),BigDecimalWithContext.getMC()));
//        }

        return outValue;
    }
}
